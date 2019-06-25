import pickle as pkl
import numpy as np
import copy
from collections import Counter
import tensorflow as tf
from data_apis.corpus import SWDADialogCorpus

# constants
tf.app.flags.DEFINE_string("word2vec_path", None, "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("api_dir", "data/weather_data/api_simdial_weather.pkl", "api dir")
tf.app.flags.DEFINE_string("work_dir", "working", "Experiment results directory.")
tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_string("test_path", "run1500783422", "the dir to load checkpoint for forward only")
tf.app.flags.DEFINE_string("result_path", "data/results.pkl", "the dir to results")
tf.app.flags.DEFINE_string("use_test_batch", True, "use test batch during testing")
FLAGS = tf.app.flags.FLAGS

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

with open(FLAGS.api_dir, "r") as fh:
    api2 = pkl.load(fh)

with open(FLAGS.result_path, "r") as fh:
    results = pkl.load(fh)

with open(FLAGS.result_path+".param.pkl", "r") as fh:
    fetch_results = pkl.load(fh)

def id_to_sent(id_to_vocab, ids):
    sent = []
    for id in ids:
        if id:
            if id_to_vocab[id] != '<s>' and id_to_vocab[id] != '</s>':
                sent.append(id_to_vocab[id])
        else:
            break
    return " ".join(sent)

def id_to_probs(probs, ids, id_to_vocab, SOFTMAX=False):
    if SOFTMAX:
        probs = softmax(probs)
    else:
        pass

    product = 1
    for id in ids:
        if id_to_vocab[id] == '</s>':
            break
        elif id_to_vocab[id] == '<s>':
            pass
        elif id:
            product *= probs[id]
        else:
            print("")
            raise Exception("id is empty!")
    return product



def id_to_log_probs(probs, ids, id_to_vocab, SOFTMAX=False):
    if SOFTMAX:
        probs = softmax(probs)
    else:
        pass

    sum = 0
    for id in ids:
        if id_to_vocab[id] == '</s>':
            break
        elif id_to_vocab[id] == '<s>':
            pass
        elif id:
            sum += np.log(probs[id])
        else:
            print("")
            raise Exception("id is empty!")
    return sum


converted_labels = []
converted_sents = []
conv_probs = []
for batch_i in xrange(len(results)):
    probs = results[batch_i][1]# [0]
    trans_probs = results[batch_i][2]
    bow_logits1 = results[batch_i][3]
    bow_logits2 = results[batch_i][4]
    usr_sents = results[batch_i][0][0]
    sys_sents = results[batch_i][0][1]
    for i in xrange(16):
        this_dialog_labels = []
        this_dialog_sents = []
        prev_label = -1
        this_conv_prob = 1
        for turn_j in xrange(probs.shape[1]):
            if probs[i, turn_j].any():
                label = probs[i, turn_j].argmax()
                # cur_trans_probs = tran
                usr_tokens = id_to_sent(api2.id_to_vocab, usr_sents[i, turn_j])
                sys_tokens = id_to_sent(api2.id_to_vocab, sys_sents[i, turn_j])
                usr_prob = id_to_log_probs(bow_logits1[i, turn_j], usr_sents[i, turn_j], api2.id_to_vocab, SOFTMAX=True)
                sys_prob = id_to_log_probs(bow_logits2[i, turn_j], sys_sents[i, turn_j], api2.id_to_vocab, SOFTMAX=True)

                this_dialog_labels += [label]
                this_dialog_sents += [[usr_tokens, sys_tokens]]
                this_turn_prob = usr_prob + sys_prob
                print(this_turn_prob)
                this_conv_prob += this_turn_prob
        conv_probs.append(this_conv_prob)
        converted_labels.append(this_dialog_labels)
        converted_sents.append(this_dialog_sents)




def get_state_sents(state, last_n=3, sys_side=1):
    zero_state_sents = []
    for i in xrange(len(converted_sents)):
        for j, label in enumerate(converted_labels[i]):
            if label == state:
                if converted_sents[i][j][1]:
                    last_n_sents = [converted_sents[i][j - i_last_n][sys_side] for i_last_n in xrange(last_n) if
                                    (j - i_last_n) >= 0]
                    last_n_sents = last_n_sents[::-1]
                    last_n_sents = "\n ".join(last_n_sents)

                    zero_state_sents.append(last_n_sents)
    return zero_state_sents

N_STATE = 10
sents_by_state = []
for i in xrange(N_STATE):
    sents_by_state.append(get_state_sents(i, sys_side=0, last_n=1))
sents_by_state_sys = []
for i in xrange(N_STATE):
    sents_by_state_sys.append(get_state_sents(i, sys_side=1, last_n=1))
WITH_START = True
if WITH_START:
    sents_by_state = [['START']] + sents_by_state
    sents_by_state_sys = [['START']] + sents_by_state_sys

transition_count = np.zeros((N_STATE, N_STATE))

for labels in converted_labels:
    # origin = 0
    for i in xrange(len(labels)-1):
        #dest = l
        print(i)
        transition_count[labels[i], labels[i+1]] += 1
        #origin = dest
    #transition_prob[origin, 11] += 1

transition_prob = np.eye((N_STATE, N_STATE))
for i in xrange(N_STATE):
    transition_prob[i] = transition_count[i]/transition_count[i].sum()


# direct transition only, for direct transition, the transition probs are from the fetch_results from the model
DIRECT_TRANSITION = False
if DIRECT_TRANSITION:
    label_i_list = np.eye(N_STATE, N_STATE)
    for i in range(N_STATE):
        label_i_list[i][i] = 1
        print(label_i_list)
    label_i_list = np.vstack([[1]*N_STATE,label_i_list])

    prob_list = []
    for i in range(0, N_STATE+1):
        tmp_prob = np.matmul(np.matmul(np.matmul(label_i_list[i], fetch_results[0]) + fetch_results[1], fetch_results[2])
                             + fetch_results[3], fetch_results[4]) + fetch_results[5]
        prob_list.append(softmax(tmp_prob))

    transition_prob = prob_list

Counter(sents_by_state[0]).most_common(5)
Counter(sents_by_state[1]).most_common(5)
Counter(sents_by_state[2]).most_common(5)
Counter(sents_by_state[3]).most_common(5)
Counter(sents_by_state[4]).most_common(5)
Counter(sents_by_state[5]).most_common(5)
Counter(sents_by_state[6]).most_common(5)
Counter(sents_by_state[7]).most_common(5)
Counter(sents_by_state[8]).most_common(5)
Counter(sents_by_state[9]).most_common(5)
Counter(sents_by_state[10]).most_common(5)
Counter(sents_by_state[11]).most_common(5)

for i in xrange(N_STATE):
    print(i)
    print(Counter(sents_by_state[i]).most_common(10))
    print("\n\n\n")

states_human_interpretation = [""] * 11
# fill in your interpretation
"""states_human_interpretation[0] = "start"
states_human_interpretation[1] = "thank you. goodbye/you are welcome goodbye"
states_human_interpretation[2] = "I want to find/anything else"
states_human_interpretation[3] = "dont care value"
states_human_interpretation[4] = "you are welcome , goodbye"
states_human_interpretation[5] = "thank you good bye"
states_human_interpretation[6] = "thank you good bye/anything else"
states_human_interpretation[7] = "address and phone"
states_human_interpretation[8] = "looking for/no match"
states_human_interpretation[9] = "goodbye"
states_human_interpretation[10] = "goodbye/anything else"
"""





with open("data/results/sents_by_state", "w") as fh:
    pkl.dump(sents_by_state, fh)

with open("data/results/transition_prob.pkl", "w") as fh:
    pkl.dump(transition_prob, fh)

with open("data/results/simulated/sents_by_state", "w") as fh:
    pkl.dump(sents_by_state, fh)

with open("data/results/simulated/transition_prob.pkl", "w") as fh:
    pkl.dump(transition_prob, fh)

