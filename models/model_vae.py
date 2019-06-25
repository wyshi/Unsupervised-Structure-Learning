from __future__ import print_function
from __future__ import division

import os
import re
import sys
import time

import numpy as np
import tensorflow as tf
from utils import *
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import rnn as rnn_cell
from models.VAE_cell import VAECell
from models.dynamic_VAE import dynamic_vae

# batch_size = None
# vocab_size = 2000
# embedding_dim = 300
# vocab_size = 10000
# W_embedding = tf.get_variable("w_embedding", shape=(vocab_size, embedding_dim), dtype=tf.float32)

class BaseProbModel(object):
    """
    most basic model, multiplication of probs
    """
    def __init__(self, sess, config, api, log_dir, scope=None):
        self.sess = sess
        self.config = config
        self.n_state = config.n_state

class VRNN(object):
    """
    VRNN with gumbel-softmax
    """
    def __init__(self, sess, config, api, log_dir, scope=None):
        self.sess = sess
        self.config = config
        self.n_state = config.n_state
        self.n_vocab = len(api.vocab)
        self.cell_type = config.cell_type
        self.encoding_cell_size = config.encoding_cell_size
        self.state_cell_size = config.state_cell_size
        self.keep_prob = config.keep_prob
        self.num_layer = config.num_layer
        self.max_utt_len = config.max_utt_len
        self.scope = scope
        with_label_loss = self.config.with_label_loss

        with tf.name_scope("io"):
            self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")
            self.usr_input_sent = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len), name="user_input")
            self.sys_input_sent = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len), name="user_input")
            self.dialog_length_mask = tf.placeholder(dtype=tf.int32, shape=(None), name="dialog_length_mask")
            self.usr_full_mask = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len), name="usr_full_mask")
            self.sys_full_mask = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len), name="sys_full_mask")
            max_dialog_len = tf.shape(self.usr_input_sent)[1]

            self.learning_rate = tf.Variable(float(config.init_lr), trainable=False, name="learning_rate")
            self.learning_rate_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, config.lr_decay))
            self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")
            self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")

            if self.config.with_label_loss:
                with tf.name_scope("labeled_id"):
                    self.labeled_usr_input_sent = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len),
                                                         name="labeled_user_input") #batch_size, dialog_len, max_utt_len
                    self.labeled_sys_input_sent = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len),
                                                         name="labeled_user_input")
                    self.labeled_dialog_length_mask = tf.placeholder(dtype=tf.int32, shape=(None), name="labeled_dialog_length_mask")
                    self.labeled_usr_full_mask = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len),
                                                        name="labeled_usr_full_mask")
                    self.labeled_sys_full_mask = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len),
                                                        name="labeled_sys_full_mask")
                    self.labeled_labels = tf.placeholder(tf.int32, shape=(None, None), name="labeled_labels")

        with variable_scope.variable_scope("sent_embedding"):
            self.W_embedding = tf.get_variable("W_embedding", [self.n_vocab, config.embed_size], dtype=tf.float32)
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.n_vocab)], dtype=tf.float32,
                                         shape=[self.n_vocab, 1])
            W_embedding = self.W_embedding * embedding_mask

            usr_input_embedding = tf.nn.embedding_lookup(W_embedding, tf.reshape(self.usr_input_sent, [-1])) # (8000, 300)
            usr_input_embedding = tf.reshape(usr_input_embedding, [-1, self.max_utt_len, self.config.embed_size]) #(160, 50, 300)
            sys_input_embedding = tf.nn.embedding_lookup(W_embedding, tf.reshape(self.sys_input_sent, [-1])) # (8000, 300)
            sys_input_embedding = tf.reshape(sys_input_embedding, [-1, self.max_utt_len, self.config.embed_size]) #(160, 50, 300)

            if self.config.with_label_loss:
                labeled_usr_input_embedding = tf.nn.embedding_lookup(W_embedding,
                                                             tf.reshape(self.labeled_usr_input_sent, [-1]))  # (8000, 300)
                labeled_usr_input_embedding = tf.reshape(labeled_usr_input_embedding,
                                                 [-1, self.max_utt_len, self.config.embed_size])  # (160, 50, 300)
                labeled_sys_input_embedding = tf.nn.embedding_lookup(W_embedding,
                                                             tf.reshape(self.labeled_sys_input_sent, [-1]))  # (8000, 300)
                labeled_sys_input_embedding = tf.reshape(labeled_sys_input_embedding,
                                                 [-1, self.max_utt_len, self.config.embed_size])  # (160, 50, 300)

        with variable_scope.variable_scope("sent_level"):
            self.encoding_cell = self.get_rnncell(self.cell_type,
                                                   self.encoding_cell_size,
                                                   self.keep_prob,
                                                   num_layer=self.num_layer)
            usr_input_embedding, usr_sent_size = get_rnn_encode(usr_input_embedding, self.encoding_cell, scope="sent_embedding_rnn")
            sys_input_embedding, sys_sent_size = get_rnn_encode(sys_input_embedding, self.encoding_cell, scope="sent_embedding_rnn", reuse=True)

            usr_input_embedding = tf.reshape(usr_input_embedding[1], [-1, max_dialog_len, usr_sent_size[0]])
            sys_input_embedding = tf.reshape(sys_input_embedding[1], [-1, max_dialog_len, sys_sent_size[0]])

            if self.config.with_label_loss:
                labeled_usr_input_embedding, labeled_usr_sent_size = get_rnn_encode(labeled_usr_input_embedding,
                                                                                    self.encoding_cell,
                                                                                    scope="sent_embedding_rnn",
                                                                                    reuse=True)
                labeled_sys_input_embedding, labeled_sys_sent_size = get_rnn_encode(labeled_sys_input_embedding,
                                                                                    self.encoding_cell,
                                                                                    scope="sent_embedding_rnn",
                                                                                    reuse=True)

                labeled_usr_input_embedding = tf.reshape(labeled_usr_input_embedding[1], [-1, max_dialog_len, labeled_usr_sent_size[0]])
                labeled_sys_input_embedding = tf.reshape(labeled_sys_input_embedding[1], [-1, max_dialog_len, labeled_sys_sent_size[0]])

            if config.keep_prob < 1.0:
                usr_input_embedding = tf.nn.dropout(usr_input_embedding, config.keep_prob)
                sys_input_embedding = tf.nn.dropout(sys_input_embedding, config.keep_prob)
                if self.config.with_label_loss:
                    labeled_usr_input_embedding = tf.nn.dropout(labeled_usr_input_embedding, config.keep_prob)
                    labeled_sys_input_embedding = tf.nn.dropout(labeled_sys_input_embedding, config.keep_prob)

            joint_embedding = tf.concat([usr_input_embedding, sys_input_embedding], 2, "joint_embedding") # (batch, dialog_len, embedding_size*2) (16, 10, 400)
            if self.config.with_label_loss:
                labeled_joint_embedding = tf.concat([labeled_usr_input_embedding, labeled_sys_input_embedding], 2,
                                            "labeled_joint_embedding")  # (batch, dialog_len, embedding_size*2) (16, 10, 400)

        with variable_scope.variable_scope("state_level"):
            usr_state_vocab_matrix = tf.get_variable("usr_state_vocab_distribution", [self.n_state, self.n_vocab],
                                                     dtype=tf.float32, initializer=tf.random_uniform_initializer())
            sys_state_vocab_matrix = tf.get_variable("sys_state_vocab_distribution", [self.n_state, self.n_vocab],
                                                     dtype=tf.float32, initializer=tf.random_uniform_initializer())
            self.usr_state_vocab_matrix = tf.nn.softmax(usr_state_vocab_matrix, -1)
            self.sys_state_vocab_matrix = tf.nn.softmax(sys_state_vocab_matrix, -1)


            self.state_cell = self.get_rnncell(self.cell_type,
                                               self.encoding_cell_size,
                                               self.keep_prob,
                                               num_layer=self.num_layer,
                                               activation=tf.nn.tanh)
            self.VAE_cell = VAECell(num_units=300,
                                    state_cell=self.state_cell,
                                    num_zt=self.config.n_state,
                                    vocab_size=self.n_vocab,
                                    max_utt_len=self.max_utt_len,
                                    config=config,
                                    use_peepholes=False, cell_clip=None,
                                    initializer=None, num_proj=None, proj_clip=None,
                                    num_unit_shards=None, num_proj_shards=None,
                                    forget_bias=1.0, state_is_tuple=True,
                                    activation=None, reuse=None, name=None)

            # dec_input_embeding = placeholder(float32, (16, max_dialog_len, 50, 300))
            # dec_seq_lens = placeholder(float32, (16, max_dialog_len))
            # output_tokens = ((16, max_dialog_len, 50), int32)
            # sequence_length = (tf.int32, (16))

            #print("before embedding")
            #print(W_embedding)
            #print(self.usr_input_sent)
            dec_input_embedding_usr = tf.nn.embedding_lookup(W_embedding, self.usr_input_sent) # (16, 10, 50, 300)
            dec_input_embedding_sys = tf.nn.embedding_lookup(W_embedding, self.sys_input_sent) # (16, 10, 50, 300)
            #print("embedding")
            dec_input_embedding = [dec_input_embedding_usr, dec_input_embedding_sys]
            #print(dec_input_embedding)

            dec_seq_lens_usr = tf.reduce_sum(tf.sign(self.usr_full_mask), 2)
            dec_seq_lens_sys = tf.reduce_sum(tf.sign(self.sys_full_mask), 2)
            dec_seq_lens = [dec_seq_lens_usr, dec_seq_lens_sys]

            output_tokens_usr = self.usr_input_sent
            output_tokens_sys = self.sys_input_sent
            output_tokens = [output_tokens_usr, output_tokens_sys]

            if self.config.with_label_loss:
                labeled_dec_input_embedding_usr = tf.nn.embedding_lookup(W_embedding,
                                                                 self.labeled_usr_input_sent)  # (16, 10, 50, 300)
                labeled_dec_input_embedding_sys = tf.nn.embedding_lookup(W_embedding,
                                                                 self.labeled_sys_input_sent)  # (16, 10, 50, 300)
                labeled_dec_input_embedding = [labeled_dec_input_embedding_usr, labeled_dec_input_embedding_sys]

                labeled_dec_seq_lens_usr = tf.reduce_sum(tf.sign(self.labeled_usr_full_mask), 2)
                labeled_dec_seq_lens_sys = tf.reduce_sum(tf.sign(self.labeled_sys_full_mask), 2)
                labeled_dec_seq_lens = [labeled_dec_seq_lens_usr, labeled_dec_seq_lens_sys]

                labeled_output_tokens_usr = self.labeled_usr_input_sent
                labeled_output_tokens_sys = self.labeled_sys_input_sent
                labeled_output_tokens = [labeled_output_tokens_usr, labeled_output_tokens_sys]

            with variable_scope.variable_scope("dynamic_VAE_loss") as dynamic_vae_scope:
                self.initial_prev_z = tf.placeholder(tf.float32, (None, self.config.n_state), 'initial_prev_z')
                losses, z_ts, p_ts, bow_logits1, bow_logits2 = dynamic_vae(self.VAE_cell,
                                           joint_embedding,
                                           dec_input_embedding,
                                           dec_seq_lens,
                                           output_tokens,
                                           z_t_size=self.config.n_state,
                                           sequence_length=self.dialog_length_mask,
                                           initial_state=None,
                                           dtype=tf.float32,
                                           parallel_iterations=None,
                                           swap_memory=False,
                                           time_major=False,
                                           scope=None,
                                           initial_prev_z=self.initial_prev_z)

                if self.config.with_label_loss:
                    dynamic_vae_scope.reuse_variables()
                    labeled_losses, labeled_z_ts, labeled_pts, labeled_bow_logits1, labeled_bow_logits2 = dynamic_vae(self.VAE_cell,
                                               labeled_joint_embedding,
                                               labeled_dec_input_embedding,
                                               labeled_dec_seq_lens,
                                               labeled_output_tokens,
                                               z_t_size=self.config.n_state,
                                               sequence_length=self.labeled_dialog_length_mask,
                                               initial_state=None,
                                               dtype=tf.float32,
                                               parallel_iterations=None,
                                               swap_memory=False,
                                               time_major=False,
                                               scope=None)
                    self.labeled_z_ts = labeled_z_ts
                    self.labeled_z_ts_mask = tf.to_float(tf.sign(tf.reduce_sum(self.labeled_usr_full_mask, 2)))

                    labeled_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.labeled_z_ts, labels=self.labeled_labels)
                    labeled_loss = tf.reduce_sum(labeled_loss * self.labeled_z_ts_mask)

                    labeled_loss = labeled_loss/tf.to_float(tf.reduce_sum(self.labeled_usr_full_mask)+tf.reduce_sum(self.labeled_sys_full_mask))
                    self.labeled_loss = tf.identity(labeled_loss, name="labeled_loss")

            z_ts = tf.nn.softmax(z_ts) # (16, 10, 12)
            z_ts_mask = tf.to_float(tf.sign(tf.reduce_sum(self.usr_full_mask, 2))) # (16, 10)
            z_ts_mask = tf.expand_dims(z_ts_mask, 2) # (16, 10, 1)
            self.z_ts = z_ts * z_ts_mask
            self.p_ts = p_ts
            self.bow_logits1 = bow_logits1
            self.bow_logits2 = bow_logits2
            loss_avg = tf.reduce_sum(losses)/tf.to_float(tf.reduce_sum(self.usr_full_mask)+tf.reduce_sum(self.sys_full_mask))

            if self.config.with_label_loss:
                loss_avg = loss_avg + self.labeled_loss

            loss_avg = tf.identity(loss_avg, name="loss_average")

            self.basic_loss = loss_avg
            tf.summary.scalar("basic_loss", self.basic_loss)

            self.summary_op = tf.summary.merge_all()

            self.optimize(sess=sess, config=config, loss=self.basic_loss, log_dir=log_dir)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    @staticmethod
    def get_rnncell(cell_type, cell_size, keep_prob, activation=tf.tanh, num_layer=1):
        # thanks for this solution from @dimeldo
        cells = []
        for _ in range(num_layer):
            if cell_type == "gru":
                cell = rnn_cell.GRUCell(cell_size, activation=activation)
            else:
                cell = rnn_cell.LSTMCell(cell_size, use_peepholes=False, activation=activation, forget_bias=1.0)

            if keep_prob < 1.0:
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

            cells.append(cell)

        if num_layer > 1:
            cell = rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        else:
            cell = cells[0]

        return cell

    @staticmethod
    def print_loss(prefix, loss_names, losses, postfix):
        template = "%s "
        for name in loss_names:
            template += "%s " % name
            template += " %f "
        template += "%s"
        template = re.sub(' +', ' ', template)
        avg_losses = []
        values = [prefix]

        for loss in losses:
            values.append(np.mean(loss))
            avg_losses.append(np.mean(loss))
        values.append(postfix)

        print(template % tuple(values))
        return avg_losses

    def batch_2_feed(self, batch, global_t, use_prior, labeled_batch=None, labeled_labels=None, repeat=1):
        usr_input_sent, sys_input_sent, dialog_len_mask, usr_full_mask, sys_full_mask = batch
        if self.config.with_label_loss:
            labeled_usr_input_sent, labeled_sys_input_sent, labeled_dialog_len_mask, labeled_usr_full_mask, labeled_sys_full_mask = labeled_batch
        feed_dict = {
            self.usr_input_sent: usr_input_sent,
            self.sys_input_sent: sys_input_sent,
            self.dialog_length_mask: dialog_len_mask,
            self.usr_full_mask: usr_full_mask,
            self.sys_full_mask: sys_full_mask
        }

        if self.config.with_label_loss:
            feed_dict.update({
                self.labeled_usr_input_sent: labeled_usr_input_sent,
                self.labeled_sys_input_sent: labeled_sys_input_sent,
                self.labeled_dialog_length_mask: labeled_dialog_len_mask,
                self.labeled_usr_full_mask: labeled_usr_full_mask,
                self.labeled_sys_full_mask: labeled_sys_full_mask,
                self.labeled_labels: labeled_labels
            })

        feed_dict.update({self.initial_prev_z: np.ones(shape=(self.config.batch_size, self.config.n_state))})

        if repeat > 1:
            tiled_feed_dict = {}
            for key, val in feed_dict.items():
                if key is self.use_prior:
                    tiled_feed_dict[key] = val
                    continue
                multipliers = [1]*len(val.shape)
                multipliers[0] = repeat
                tiled_feed_dict[key] = np.tile(val, multipliers)
            feed_dict = tiled_feed_dict

        if global_t is not None:
            feed_dict[self.global_t] = global_t

        return feed_dict

    def train(self, global_t, sess, train_feed, labeled_feed=None, labeled_dial_labels=None, update_limit=5000):
        base_losses = []
        local_t = 0
        start_time = time.time()
        loss_names = ["base_loss"]
        while True:
            batch = train_feed.next_batch()
            if self.config.with_label_loss:
                labeled_batch = labeled_feed.next_batch()
            if batch is None:
                break
            if update_limit is not None and local_t >= update_limit:
                break
            if self.config.with_label_loss:
                feed_dict = self.batch_2_feed(batch=batch, labeled_batch=labeled_batch,
                                              labeled_labels=labeled_dial_labels, global_t=global_t, use_prior=False)
            else:
                feed_dict = self.batch_2_feed(batch=batch, labeled_batch=None,
                                              labeled_labels=labeled_dial_labels, global_t=global_t, use_prior=False)

            _, sum_op, base_loss = sess.run([self.train_ops,
                                             self.summary_op,
                                             self.basic_loss],
                                            feed_dict)
            self.train_summary_writer.add_summary(sum_op, global_t)
            base_losses.append(base_loss)

            global_t += 1
            local_t += 1
            if local_t % (train_feed.num_batch // 10) == 0:
                # kl_w = sess.run(self.kl_w, {self.global_t: global_t}
                self.print_loss("%.2f" % (train_feed.ptr / float(train_feed.num_batch)),
                                loss_names, [base_losses], postfix='kl_w')

        # finish epoch!
        epoch_time = time.time() - start_time
        avg_losses = self.print_loss("Epoch Done", loss_names,
                                     [base_losses],
                                     "step time %.4f" % (epoch_time / train_feed.num_batch))

        return global_t, avg_losses[0]

    def valid(self, name, sess, valid_feed, labeled_feed=None, labeled_labels=None):
        elbo_losses = []

        while True:
            batch = valid_feed.next_batch()
            if self.config.with_label_loss:
                labeled_batch = labeled_feed.next_batch()
            if batch is None:
                break
            if self.config.with_label_loss:
                feed_dict = self.batch_2_feed(batch=batch, labeled_batch=labeled_batch, labeled_labels=labeled_labels, global_t=None, use_prior=False, repeat=1)
            else:
                feed_dict = self.batch_2_feed(batch=batch, labeled_batch=None, labeled_labels=None, global_t=None, use_prior=False, repeat=1)

            elbo_loss = sess.run(self.basic_loss,
                                 feed_dict)
            elbo_losses.append(elbo_loss)

        avg_losses = self.print_loss(name, ['elbo_losses valid'], [elbo_losses], "")

        return avg_losses[0]

    def get_zt(self, global_t, sess, train_feed, update_limit=5000, labeled_feed=None, labeled_labels=None):
        local_t = 0
        start_time = time.time()
        results = []
        i_batch = 0
        while i_batch < train_feed.num_batch:
            # print(train_feed.num_batch)
            batch = train_feed.next_batch()
            if self.config.with_label_loss:
                labeled_batch = labeled_feed.next_batch()
            if batch is None:
                break
            if update_limit is not None and local_t >= update_limit:
                break

            if self.config.with_label_loss:
                feed_dict = self.batch_2_feed(batch=batch, labeled_batch=labeled_batch,
                                              labeled_labels=labeled_labels, global_t=None, use_prior=False, repeat=1)
            else:
                feed_dict = self.batch_2_feed(batch=batch, labeled_batch=None,
                                              labeled_labels=labeled_labels, global_t=None, use_prior=False, repeat=1)

            fetch_list = ['model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_1/weights:0',
                          'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_1/biases:0',
                          'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_2/weights:0',
                          'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_2/biases:0',
                          'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/fully_connected/weights:0',
                          'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/fully_connected/biases:0']
            z_ts, p_ts, bow_logits1, bow_logits2, w1, b1, w2, b2, w3, b3 = sess.run([self.z_ts,
                                    self.p_ts, self.bow_logits1, self.bow_logits2] + fetch_list,
                            feed_dict)
            # print(w_1)

            global_t += 1
            local_t += 1
            i_batch += 1
            result = [batch, z_ts, p_ts, bow_logits1, bow_logits2]
            results.append(result)
            # if local_t % (train_feed.num_batch // 10) == 0:
            #     # kl_w = sess.run(self.kl_w, {self.global_t: global_t}
            #     print("%.2f" % (train_feed.ptr / float(train_feed.num_batch)))


        epoch_time = time.time() - start_time
        return results, [w1, b1, w2, b2, w3, b3]

    def get_log_prob(self, global_t, sess, train_feed, transition_prob, update_limit=5000, labeled_feed=None, labeled_labels=None):
        # reconstruct the decoder to get self.bow_logits1 and self.bow_logits2
        local_t = 0
        start_time = time.time()
        results = []
        i_batch = 0
        while i_batch < train_feed.num_batch:
            # print(train_feed.num_batch)
            batch = train_feed.next_batch()
            if self.config.with_label_loss:
                labeled_batch = labeled_feed.next_batch()
            if batch is None:
                break
            if update_limit is not None and local_t >= update_limit:
                break

            if self.config.with_label_loss:
                feed_dict = self.batch_2_feed(batch=batch, labeled_batch=labeled_batch,
                                              labeled_labels=labeled_labels, global_t=None, use_prior=False, repeat=1)
            else:
                feed_dict = self.batch_2_feed(batch=batch, labeled_batch=None,
                                              labeled_labels=labeled_labels, global_t=None, use_prior=False, repeat=1)

            fetch_list = ['model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_1/weights:0',
                          'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_1/biases:0',
                          'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_2/weights:0',
                          'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/Stack/fully_connected_2/biases:0',
                          'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/fully_connected/weights:0',
                          'model/state_level/dynamic_VAE_loss/VAE/one_step_VAE/priorNetwork/fully_connected/biases:0']
            z_ts, p_ts, w1, b1, w2, b2, w3, b3 = sess.run([self.z_ts,
                                    self.p_ts] + fetch_list,
                            feed_dict)
            # print(w_1)

            global_t += 1
            local_t += 1
            i_batch += 1
            result = [batch, z_ts, p_ts]
            results.append(result)
            if local_t % (train_feed.num_batch // 10) == 0:
                # kl_w = sess.run(self.kl_w, {self.global_t: global_t}
                print("%.2f" % (train_feed.ptr / float(train_feed.num_batch)))


        epoch_time = time.time() - start_time
        return results, [w1, b1, w2, b2, w3, b3]

    def print_model_stats(self, tvars):
        total_parameters = 0
        for variable in tvars:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            print("Trainable %s with %d parameters" % (variable.name, variable_parametes))
            total_parameters += variable_parametes
        print("Total number of trainable parameters is %d" % total_parameters)

    def optimize(self, sess, config, loss, log_dir):
        if log_dir is None:
            return
        # optimization
        if self.scope is None:
            tvars = tf.trainable_variables()
        else:
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        grads = tf.gradients(loss, tvars)
        if config.grad_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, tf.constant(config.grad_clip))
        # add gradient noise
        if config.grad_noise > 0:
            grad_std = tf.sqrt(config.grad_noise / tf.pow(1.0 + tf.to_float(self.global_t), 0.55))
            grads = [g + tf.truncated_normal(tf.shape(g), mean=0.0, stddev=grad_std) for g in grads]

        if config.op == "adam":
            print("Use Adam")
            optimizer = tf.train.AdamOptimizer(config.init_lr)
        elif config.op == "rmsprop":
            print("Use RMSProp")
            optimizer = tf.train.RMSPropOptimizer(config.init_lr)
        else:
            print("Use SGD")
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.train_ops = optimizer.apply_gradients(zip(grads, tvars))
        self.print_model_stats(tvars)
        train_log_dir = os.path.join(log_dir, "checkpoints")
        print("Save summary to %s" % log_dir)
        self.train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)






















