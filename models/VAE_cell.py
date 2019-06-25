from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from models.seq2seq import dynamic_rnn_decoder
import models.decoder_fn_lib as decoder_fn_lib
from tensorflow.contrib import rnn as rnn_cell
from tensorflow.contrib.rnn import OutputProjectionWrapper
from tensorflow.contrib import layers
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.framework import ops

slim = tf.contrib.slim

def compute_weighted_loss(losses, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES):
    with tf.name_scope(scope, "weighted_loss", (losses, weights)):
        with tf.control_dependencies((
            weights_broadcast_ops.assert_broadcastable(weights, losses),)):
            losses = tf.convert_to_tensor(losses)
            input_type = losses.dtype
            losses = tf.to_float(losses)
            weights = tf.to_float(weights)
            weighted_losses = tf.multiply(losses, weights)
            loss = weighted_losses

            loss = tf.cast(loss, input_type)
            util.add_loss(loss, loss_collection)
            return loss

# thanks for the implementation at https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature), y/temperature


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y, logits = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y, logits


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


class VAECell(object):
    """

    """

    def __init__(self, num_units, state_cell, vocab_size, max_utt_len, config, num_zt=10,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None, reuse=None, name=None, dtype=None):

        self._state_is_tuple = state_is_tuple
        self.num_zt = num_zt
        self.tau = tf.Variable(5.0, name="temperature")
        self.vocab_size = vocab_size
        self.max_utt_len = max_utt_len
        self.config = config
        if self.config.word_weights:
            self.weights = tf.constant(self.config.word_weights)
        else:
            self.weights = self.config.word_weights
        self.decoder_cell_1 = self.get_rnncell('lstm', 200+num_zt, keep_prob=self.config.keep_prob)
        self.decoder_cell_1 = OutputProjectionWrapper(self.decoder_cell_1, self.vocab_size)
        self.decoder_cell_2 = self.get_rnncell('lstm', 2*(200+num_zt), keep_prob=self.config.keep_prob)
        self.decoder_cell_2 = OutputProjectionWrapper(self.decoder_cell_2, self.vocab_size)
        self.state_cell = state_cell

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

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self.state_cell.zero_state(batch_size, dtype)

    def build(self):
        pass

    def __call__(self, inputs, state, dec_input_embedding, dec_seq_lens, output_tokens, forward=False, prev_z_t=None):
        """

        :param inputs: [batch, sentence_encoding_size*2]
        :param state:
        :param dec_input_embedding:
        :param dec_seq_lens:
        :param output_tokens:
        :param forward:
        :return:
        """
        if self._state_is_tuple:
            (c_prev, h_prev) = state
        if self.config.with_direct_transition:
            assert prev_z_t is not None

        print(type(dec_input_embedding[0]))
        with vs.variable_scope("one_step_VAE") as vaeScope:
            try:
                test_reuse_variable = tf.Variable(5.0, name="test_reuse")
            except ValueError:
                vaeScope.reuse_variables()

            with vs.variable_scope("encoderNetwork") as encScope:
                enc_inputs = tf.concat([h_prev, inputs], 1, "encNet_inputs") # [batch, sent_encoding_size*2 + state_cell_size]
                net = slim.stack(enc_inputs, slim.fully_connected, [400, 200])
                if self.config.keep_prob < 1.0:
                    net = tf.nn.dropout(net, self.config.keep_prob)
                logits_z = slim.fully_connected(net, self.num_zt, activation_fn=None) # [batch, num_zt]
                q_z = tf.nn.softmax(logits_z)
                log_q_z = tf.log(q_z+1e-20)

                z_samples, logits_z_samples = gumbel_softmax(logits_z, self.tau, hard=False) # [batch, num_zt]
                print("z_samples")
                print(z_samples)

            with vs.variable_scope("decoderNetwork") as decScope:
                net2 = slim.stack(z_samples, slim.fully_connected, [200, 200]) # phi_z_feature_extraction
                if self.config.keep_prob < 1.0:
                    net2 = tf.nn.dropout(net2, self.config.keep_prob)
                dec_input_1 = tf.concat([h_prev, net2], 1, "decNet_inputs_1") # [batch, state_cell_size + 200]

                dec_init_state_1 = tf.contrib.rnn.LSTMStateTuple(dec_input_1, dec_input_1)

                if not forward:
                    selected_attribute_embedding = None
                    loop_func_1 = decoder_fn_lib.context_decoder_fn_train(dec_init_state_1, selected_attribute_embedding)
                    # dec_input_embedding = embedding_ops.embedding_lookup(embedding, self.output_tokens)
                    # dec_input_embedding = tf.placeholder(tf.float32, (16, 50, 300))
                    # dec_seq_lens = tf.placeholder(tf.int32, (16))

                    dec_input_embedding[0] = dec_input_embedding[0][:, 0:-1, :]
                    dec_input_embedding[1] = dec_input_embedding[1][:, 0:-1, :]
                    dec_outs_1, final_state_1 = dynamic_rnn_decoder(self.decoder_cell_1,
                                                                    loop_func_1,
                                                                    inputs=dec_input_embedding[0],
                                                                    sequence_length=dec_seq_lens[0]-1,
                                                                    scope_name="dynamic_rnn_decoder_1")

                    dec_input_2_c = tf.concat([dec_input_1, final_state_1[0]], 1, 'decNet_inputs_2_c') # [batch, state_cell_size + 200]
                    dec_input_2_h = tf.concat([dec_input_1, final_state_1[1]], 1, 'decNet_inputs_2_h') # [batch, state_cell_size + 200]

                    dec_init_state_2 = tf.contrib.rnn.LSTMStateTuple(dec_input_2_c, dec_input_2_h)
                    loop_func_2 = decoder_fn_lib.context_decoder_fn_train(dec_init_state_2, selected_attribute_embedding)
                    dec_outs_2, final_state_2 = dynamic_rnn_decoder(self.decoder_cell_2,
                                                                    loop_func_2,
                                                                    inputs=dec_input_embedding[1],
                                                                    sequence_length=dec_seq_lens[1]-1,
                                                                    scope_name="dynamic_rnn_decoder_2")

                    # for computing BOW loss
                    if self.config.with_bow_loss:
                        bow_fc1 = layers.fully_connected(dec_input_1, 400, activation_fn=tf.tanh, scope="bow_fc1")
                        if self.config.keep_prob < 1.0:
                            bow_fc1 = tf.nn.dropout(bow_fc1, self.config.keep_prob)
                        self.bow_logits1 = layers.fully_connected(bow_fc1, self.vocab_size, activation_fn=None,
                                                                  scope="bow_project1")
                        print("self.bow_logits[1]")#(None, vocab_size), None is batch size
                        print(self.bow_logits1)
                        # sys.exit()

                        bow_fc2 = layers.fully_connected(dec_input_2_h, 400, activation_fn=tf.tanh, scope="bow_fc2")
                        if self.config.keep_prob < 1.0:
                            bow_fc2 = tf.nn.dropout(bow_fc2, self.config.keep_prob)
                        self.bow_logits2 = layers.fully_connected(bow_fc2, self.vocab_size, activation_fn=None,
                                                                  scope="bow_project2")

            with vs.variable_scope("priorNetwork") as priorScope:
                if self.config.with_direct_transition:
                    net3 = slim.stack(prev_z_t, slim.fully_connected, [100, 100])
                    if self.config.keep_prob < 1.0:
                        net3 = tf.nn.dropout(net3, self.config.keep_prob)
                    p_z = slim.fully_connected(net3, self.num_zt, activation_fn=tf.nn.softmax)
                    p_z = tf.identity(p_z, name="p_z_transition")
                    log_p_z = tf.log(p_z+1e-20) # equation 5

                else:
                    net3 = slim.stack(h_prev, slim.fully_connected, [100, 100])
                    if self.config.keep_prob < 1.0:
                        net3 = tf.nn.dropout(net3, self.config.keep_prob)
                    p_z = slim.fully_connected(net3, self.num_zt, activation_fn=tf.nn.softmax)
                    log_p_z = tf.log(p_z+1e-20) # equation 5

            with vs.variable_scope("recurrence") as recScope:
                recur_input = tf.concat([net2, inputs], 1, 'recNet_inputs') # [batch, 600]
                next_state = self.state_cell(inputs=recur_input, state=state)

            with vs.variable_scope("loss"):
                self.output_tokens = output_tokens

                labels_1 = output_tokens[0][:, 1:]
                label_mask_1 = tf.to_float(tf.sign(labels_1))
                labels_2 = output_tokens[1][:, 1:]
                label_mask_2 = tf.to_float(tf.sign(labels_2))

                rc_loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs_1, labels=labels_1)
                if True:
                    if self.weights is not None:
                        weights = tf.gather(self.weights, labels_1)
                        rc_loss_1 = compute_weighted_loss(rc_loss_1, weights=weights)
                rc_loss_1 = tf.reduce_sum(rc_loss_1 * label_mask_1, reduction_indices=1)

                rc_loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs_2, labels=labels_2)
                if True:
                    if self.weights is not None:
                        rc_loss_2 = compute_weighted_loss(rc_loss_2, weights=weights)
                rc_loss_2 = tf.reduce_sum(rc_loss_2 * label_mask_2, reduction_indices=1)

                kl_tmp = (log_q_z - log_p_z) * q_z
                kl_tmp = tf.reduce_sum(kl_tmp, reduction_indices=1)

                if self.config.with_BPR:
                    q_z_prime = tf.reduce_mean(q_z, 0)
                    log_q_z_prime = tf.log(q_z_prime + 1e-20) # equation 9

                    p_z_prime = tf.reduce_mean(p_z, 0)
                    log_p_z_prime = tf.log(p_z_prime + 1e-20)

                    kl_bpr = (log_q_z_prime - log_p_z_prime) * q_z_prime
                    kl_bpr = tf.reduce_sum(kl_bpr)
                    infered_batch_size = tf.shape(inputs)[0]
                    kl_bpr = tf.div(kl_bpr, tf.to_float(infered_batch_size))

                if not self.config.with_BPR:
                    elbo_t = rc_loss_1 + rc_loss_2 + kl_tmp
                else:
                    elbo_t = rc_loss_1 + rc_loss_2 + kl_bpr

                # BOW_loss
                if self.config.with_bow_loss:
                    tile_bow_logits1 = tf.tile(tf.expand_dims(self.bow_logits1, 1), (1, self.config.max_utt_len - 1, 1))
                    print("tile_bow_logits1")
                    print(tile_bow_logits1)

                    print("labels_1")
                    print(labels_1)
                    bow_loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tile_bow_logits1, labels=labels_1) * label_mask_1
                    if True:
                        if self.weights is not None:
                            weights = tf.gather(self.weights, labels_1)
                            bow_loss1 = compute_weighted_loss(bow_loss1, weights=weights)
                    print("bow_loss1")
                    print(bow_loss1)


                    bow_loss1 = tf.reduce_sum(bow_loss1, reduction_indices=1)
                    print("bow_loss1")
                    print(bow_loss1)

                    tile_bow_logits2 = tf.tile(tf.expand_dims(self.bow_logits2, 1), (1, self.config.max_utt_len - 1, 1))
                    bow_loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tile_bow_logits2, labels=labels_2) * label_mask_2
                    if True:
                        if self.weights is not None:
                            bow_loss2 = compute_weighted_loss(bow_loss2, weights=weights)
                    bow_loss2 = tf.reduce_sum(bow_loss2, reduction_indices=1)

                    print("bow_loss1")
                    print(bow_loss1)
                    #sys.exit()
                    elbo_t = elbo_t + self.config.bow_loss_weight*(bow_loss1 + bow_loss2)

                elbo_t = tf.expand_dims(elbo_t, 1)
        print("z_samples")
        print(z_samples)
        print("\n\n\n")



        return elbo_t, logits_z_samples, next_state[1], p_z, self.bow_logits1, self.bow_logits2


