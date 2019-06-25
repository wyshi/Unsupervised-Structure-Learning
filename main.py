# Original work Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
# Modified work Copyright 2018 Weiyan Shi.
from __future__ import print_function
from __future__ import division

import os
import time
import pickle as pkl

import numpy as np
import tensorflow as tf
from beeprint import pp

from config_utils import KgCVAEConfig as Config
from data_apis.corpus import SWDADialogCorpus
from data_apis.data_utils import SWDADataLoader
from models.model_vae import VRNN

# constants
tf.app.flags.DEFINE_string("word2vec_path", None, "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("data_dir", "data/data.pkl", "Raw data directory.")
tf.app.flags.DEFINE_string("work_dir", "working", "Experiment results directory.")
tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_string("test_path", "run1500783422", "the dir to load checkpoint for forward only")
tf.app.flags.DEFINE_string("result_path", "data/results.pkl", "the dir to results")
tf.app.flags.DEFINE_string("use_test_batch", True, "use test batch during testing")
tf.app.flags.DEFINE_integer("n_state", 10, "number of states")
tf.app.flags.DEFINE_bool("with_direct_transition", False, "number of states")
tf.app.flags.DEFINE_bool("with_word_weights", False, "number of states")
FLAGS = tf.app.flags.FLAGS


def main():
    # config for training
    config = Config()

    # config for validation
    valid_config = Config()
    # valid_config.keep_prob = 1.0
    # valid_config.dec_keep_prob = 1.0
    # valid_config.batch_size = 60

    # configuration for testing
    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.dec_keep_prob = 1.0
    test_config.batch_size = 1

    config.n_state = FLAGS.n_state
    valid_config.n_state = FLAGS.n_state
    test_config.n_state = FLAGS.n_state

    config.with_direct_transition = FLAGS.with_direct_transition
    valid_config.with_direct_transition = FLAGS.with_direct_transition
    test_config.with_direct_transition = FLAGS.with_direct_transition

    config.with_word_weights = FLAGS.with_word_weights
    valid_config.with_word_weights = FLAGS.with_word_weights
    test_config.with_word_weights = FLAGS.with_word_weights

    pp(config)

    print(config.n_state)
    print(config.with_direct_transition)
    print(config.with_word_weights)
    # get data set
    # api = SWDADialogCorpus(FLAGS.data_dir, word2vec=FLAGS.word2vec_path, word2vec_dim=config.embed_size)
    with open(config.api_dir, "r") as fh:
        api = pkl.load(fh)
    dial_corpus = api.get_dialog_corpus()
    if config.with_label_loss:
        labeled_dial_labels = api.get_state_corpus(config.max_dialog_len)['labeled']
    # meta_corpus = api.get_meta_corpus()

    # train_meta, valid_meta, test_meta = meta_corpus.get("train"), meta_corpus.get("valid"), meta_corpus.get("test")
    train_dial, labeled_dial, test_dial = dial_corpus.get("train"), dial_corpus.get("labeled"), dial_corpus.get("test")

    # convert to numeric input outputs that fits into TF models
    train_feed = SWDADataLoader("Train", train_dial, config)
    # valid_feed = SWDADataLoader("Valid", valid_dial, valid_meta, config)
    test_feed = SWDADataLoader("Test", test_dial, config)
    if config.with_label_loss:
        labeled_feed = SWDADataLoader("Labeled", labeled_dial, config, labeled=True)
    valid_feed = test_feed

    if FLAGS.forward_only or FLAGS.resume:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.test_path)
    else:
        log_dir = os.path.join(FLAGS.work_dir, "run"+str(int(time.time())))

    # begin training
    with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
        initializer = tf.random_uniform_initializer(-1.0 * config.init_w, config.init_w)
        scope = "model"
        with tf.variable_scope(scope, reuse=None, initializer=initializer):
            model = VRNN(sess, config, api, log_dir=None if FLAGS.forward_only else log_dir, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            valid_model = VRNN(sess, valid_config, api, log_dir=None, scope=scope)
        #with tf.variable_scope(scope, reuse=True, initializer=initializer):
        #    test_model = KgRnnCVAE(sess, test_config, api, log_dir=None, forward=True, scope=scope)

        print("Created computation graphs")
        if api.word2vec is not None and not FLAGS.forward_only:
            print("Loaded word2vec")
            sess.run(model.W_embedding.assign(np.array(api.word2vec)))

        # write config to a file for logging
        if not FLAGS.forward_only:
            with open(os.path.join(log_dir, "run.log"), "wb") as f:
                f.write(pp(config, output=False))

        # create a folder by force
        ckp_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)

        ckpt = tf.train.get_checkpoint_state(ckp_dir)
        print("Created models with fresh parameters.")
        sess.run(tf.global_variables_initializer())

        if ckpt:
            print("Reading dm models parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            #print([str(op.name) for op in tf.get_default_graph().get_operations()])
            print([str(v.name) for v in tf.global_variables()])
            import sys
            # sys.exit()

        if not FLAGS.forward_only:
            dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__+ ".ckpt")
            global_t = 1
            patience = config.n_epoch  # wait for at least 10 epoch before stop
            dev_loss_threshold = np.inf
            best_dev_loss = np.inf
            for epoch in range(config.max_epoch):
                print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))

                # begin training
                if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                    train_feed.epoch_init(config.batch_size, shuffle=True)

                if config.with_label_loss:
                    labeled_feed.epoch_init(len(labeled_dial), shuffle=False)
                else:
                    labeled_feed = None
                    labeled_dial_labels = None
                global_t, train_loss = model.train(global_t, sess, train_feed, labeled_feed, labeled_dial_labels, update_limit=config.update_limit)

                # begin validation
                valid_feed.epoch_init(config.batch_size, shuffle=False)
                valid_loss = valid_model.valid("ELBO_VALID", sess, valid_feed, labeled_feed, labeled_dial_labels)
                """
                test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
                                     test_config.step_size, shuffle=True, intra_shuffle=False)
                test_model.test(sess, test_feed, num_batch=5)
                """

                done_epoch = epoch + 1
                # only save a models if the dev loss is smaller
                # Decrease learning rate if no improvement was seen over last 3 times.
                if config.op == "sgd" and done_epoch > config.lr_hold:
                    sess.run(model.learning_rate_decay_op)
                """
                if valid_loss < best_dev_loss:
                    if valid_loss <= dev_loss_threshold * config.improve_threshold:
                        patience = max(patience, done_epoch * config.patient_increase)
                        dev_loss_threshold = valid_loss

                    # still save the best train model
                    if FLAGS.save_model:
                        print("Save model!!")
                        model.saver.save(sess, dm_checkpoint_path, global_step=epoch)
                    best_dev_loss = valid_loss
                """
                # still save the best train model
                if FLAGS.save_model:
                    print("Save model!!")
                    model.saver.save(sess, dm_checkpoint_path, global_step=epoch)

                if config.early_stop and patience <= done_epoch:
                    print("!!Early stop due to run out of patience!!")
                    break
            print("Best validation loss %f" % best_dev_loss)
            print("Done training")
        else:
            # begin validation
            # begin validation
            global_t = 1
            for epoch in range(1):
                print("test-----------")
                print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))

            if not FLAGS.use_test_batch:
                    # begin training
                    if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                        train_feed.epoch_init(config.batch_size, shuffle=False)
                    if config.with_label_loss:
                        labeled_feed.epoch_init(len(labeled_dial), shuffle=False)
                    else:
                        labeled_feed = None
                        labeled_dial_labels = None
                    results, fetch_results = model.get_zt(global_t, sess, train_feed, update_limit=config.update_limit,
                                           labeled_feed=labeled_feed, labeled_labels=labeled_dial_labels)
                    with open(FLAGS.result_path, "w") as fh:
                        pkl.dump(results, fh)
                    with open(FLAGS.result_path+".param.pkl", "w") as fh:
                        pkl.dump(fetch_results, fh)
            else:
                print("use_test_batch")
                # begin training
                valid_feed.epoch_init(config.batch_size, shuffle=False)

                if config.with_label_loss:
                    labeled_feed.epoch_init(len(labeled_dial), shuffle=False)
                else:
                    labeled_feed = None
                    labeled_dial_labels = None
                results, fetch_results = model.get_zt(global_t, sess, valid_feed, update_limit=config.update_limit,
                                                      labeled_feed=labeled_feed, labeled_labels=labeled_dial_labels)
                with open(FLAGS.result_path, "w") as fh:
                    pkl.dump(results, fh)
                with open(FLAGS.result_path + ".param.pkl", "w") as fh:
                    pkl.dump(fetch_results, fh)

            """
            valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            valid_model.valid("ELBO_VALID", sess, valid_feed)

            test_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            valid_model.valid("ELBO_TEST", sess, test_feed)

            dest_f = open(os.path.join(log_dir, "test.txt"), "wb")
            test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
                                 test_config.step_size, shuffle=False, intra_shuffle=False)
            test_model.test(sess, test_feed, num_batch=None, repeat=10, dest=dest_f)
            dest_f.close()
            """

if __name__ == "__main__":
    print(FLAGS.resume)
    print(FLAGS.forward_only)
    if FLAGS.forward_only:
        if FLAGS.test_path is None:
            print("Set test_path before forward only")
            exit(1)
    main()













