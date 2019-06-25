# Original work Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
# Modified work Copyright 2018 Weiyan Shi.
from __future__ import print_function
from __future__ import division
import pickle as pkl
import numpy as np


class KgCVAEConfig(object):
    description= None
    use_hcf = True  # use dialog act in training (if turn off kgCVAE -> CVAE)
    update_limit = 3000  # the number of mini-batch before evaluating the model

    api_dir = "data/cambridge_data/api_cambridge.pkl"#"data/api_simdial_weather.pkl"#"data/cambridge_data/api_cambridge.pkl"
    rev_vocab_dir = "data/cambridge_data/rev_vocab.pkl"#"data/weather_rev_vocab.pkl"#"data/cambridge_data/rev_vocab.pkl"

    # state variable
    n_state = 10 # the number of states
    # full_kl_step = 10000  # how many batch before KL cost weight reaches 1.0
    # dec_keep_prob = 1.0  # do we use word drop decoder [Bowman el al 2015]

    # Network general
    cell_type = "lstm"  # gru or lstm
    encoding_cell_size = 400 # size of the rnn
    state_cell_size = n_state
    embed_size = 300  # word embedding size
    # topic_embed_size = 30  # topic embedding size
    # da_embed_size = 30  # dialog act embedding size
    # cxt_cell_size = 600  # context encoder hidden size
    # sent_cell_size = 300  # utterance encoder hidden size
    # dec_cell_size = 400  # response decoder hidden size
    # backward_size = 10  # how many utterance kept in the context window
    # step_size = 1  # internal usage
    max_utt_len = 40  # max number of words in an utterance
    max_dialog_len = 10 # max number of turns in a dialog
    num_layer = 1  # number of context RNN layers

    # Optimization parameters
    op = "adam"
    grad_clip = 5.0  # gradient abs max cut
    init_w = 0.08  # uniform random from [-init_w, init_w]
    batch_size = 16  # mini-batch size
    init_lr = 0.001  # initial learning rate
    lr_hold = 1  # only used by SGD
    lr_decay = 0.6  # only used by SGD
    keep_prob = 0.6  # drop out rate
    improve_threshold = 0.996  # for early stopping
    patient_increase = 2.0  # for early stopping
    early_stop = True
    max_epoch = 60  # max number of epoch of training
    grad_noise = 0.0  # inject gradient noise?

    with_bow_loss = True
    bow_loss_weight = 0.4 # weight of the bow_loss
    n_epoch = 10
    with_label_loss = False # semi-supervised or not

    with_BPR = True

    with_direct_transition = False # direct prior transition prob

    with_word_weights = False

    if with_word_weights:
        with open(rev_vocab_dir, "r") as fh:
            rev_vocab = pkl.load(fh)

        slot_value_id_list = []
        for k, v in rev_vocab.items():
            # print(type(k))
            if ("slot_" in k) or ("value_" in k):
                #print(k)
                slot_value_id_list.append(v)

        multiply_factor = 3
        one_weight = 1.0 / (len(rev_vocab) + (multiply_factor-1)*len(slot_value_id_list))
        word_weights = [one_weight] * len(rev_vocab)
        for i in slot_value_id_list:
            word_weights[i] = multiply_factor * word_weights[i]

        sum_word_weights = np.sum(word_weights)
        # print(sum_word_weights)
        # print(type(sum_word_weights))
        assert (sum_word_weights == float(1.0))
        word_weights = list(len(rev_vocab)*np.array(word_weights))

    else:
        word_weights = None









