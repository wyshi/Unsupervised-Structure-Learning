#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
from __future__ import print_function
from __future__ import division

import pickle as pkl
from collections import Counter
import numpy as np
import nltk
import gensim

from sklearn.preprocessing import OneHotEncoder


class SWDADialogCorpus(object):
    dialog_act_id = 0
    sentiment_id = 1
    liwc_id = 2

    def __init__(self, corpus_path, max_vocab_cnt=10000, word2vec=None, word2vec_dim=300, labeled=False):
        """
        :param corpus_path: the folder that contains the SWDA dialog corpus
        """
        self._path = corpus_path
        self.word_vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        self.dialog_id = 0
        # self.meta_id = 1
        self.utt_id = 1
        self.label_id = 2
        self.labeled = labeled
        self.sil_utt = ["<s>", "<sil>", "</s>"]
        data = pkl.load(open(self._path, "rb"))
        self.train_corpus = self.process(data["train"])
        # self.valid_corpus = self.process(data["valid"])
        self.test_corpus = self.process(data["test"])
        if self.labeled:
            self.labeled_corpus = self.process(data["labeled"], labeled=True)
        self.build_vocab(max_vocab_cnt)
        self.load_word2vec()
        print("Done loading corpus")

    def process(self, data, labeled=False):
        """new_dialog: [(a, 1/0), (a,1/0)], new_meta: (a, b, topic), new_utt: [[a,b,c)"""
        """ 1 is own utt and 0 is other's utt"""
        new_dialog = []
        if labeled:
            new_labels = []
        #
        new_meta = []
        new_utts = []
        bod_utt = ["<s>", "<d>", "</s>"]
        all_lenes = []

        for l in data:
            dialog = []
            dialog_labels = []
            for turn in l:
                usr_utt = ["<s>"] + nltk.WordPunctTokenizer().tokenize(turn[1].lower()) + ["</s>"]
                sys_utt = ["<s>"] + nltk.WordPunctTokenizer().tokenize(turn[2].lower()) + ["</s>"]
                new_utts.append(usr_utt)
                new_utts.append(sys_utt)

                all_lenes.extend([len(usr_utt)])
                all_lenes.extend([len(sys_utt)])

                dialog.append([usr_utt, sys_utt])
                if labeled:
                    dialog_labels.append(turn[4])

            new_dialog.append(dialog)
            if labeled:
                new_labels.append(dialog_labels)

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lenes), float(np.mean(all_lenes))))
        if labeled:
            return new_dialog, new_utts, new_labels
        else:
            return new_dialog, new_utts

    def build_vocab(self, max_vocab_cnt):
        all_words = []
        for tokens in self.train_corpus[self.utt_id]:
            all_words.extend(tokens)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size {}, valid size {}, \n raw vocab size {}, vocab size {} "
              "at cut_off {} OOV rate {}".format(len(self.train_corpus), len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1], float(discard_wc) / len(all_words)))

        self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab["<unk>"]
        self.id_to_vocab = {self.rev_vocab[v]: v for v in self.rev_vocab}
        print("<d> index %d" % self.rev_vocab.get("<d>", -2))
        print("<sil> index %d" % self.rev_vocab.get("<sil>", -1))

        """
        # create topic vocab
        all_topics = []
        for a, b, topic in self.train_corpus[self.meta_id]:
            all_topics.append(topic)
        self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
        self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}
        print("%d topics in train data" % len(self.topic_vocab))

        # get dialog act labels
        all_dialog_acts = []
        for dialog in self.train_corpus[self.dialog_id]:
            all_dialog_acts.extend([feat[self.dialog_act_id] for caller, utt, feat in dialog if feat is not None])
        self.dialog_act_vocab = [t for t, cnt in Counter(all_dialog_acts).most_common()]
        self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}
        print(self.dialog_act_vocab)
        print("%d dialog acts in train data" % len(self.dialog_act_vocab))
        """

    def load_word2vec(self, binary=True):
        if self.word_vec_path is None:
            return
        raw_word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.word_vec_path, binary=binary)
        print("load w2v done")
        # clean up lines for memory efficiency
        self.word2vec = []
        oov_cnt = 0
        for v in self.vocab:
            if v not in raw_word2vec:
                oov_cnt += 1
                vec = np.random.randn(self.word2vec_dim) * 0.1
            else:
                vec = raw_word2vec[v]
            self.word2vec.append(vec)
        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))

    def get_utt_corpus(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([self.rev_vocab.get(t, self.unk_id) for t in line])
            return results
        # convert the corpus into ID
        id_train = _to_id_corpus(self.train_corpus[self.utt_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.utt_id])
        id_test = _to_id_corpus(self.test_corpus[self.utt_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_dialog_corpus(self):
        def _to_id_corpus(data):
            # results[0][1][0], dialog 0, 1st turn, 0 is usr, 1 is sys
            # results[0][1][1], dialog 0, 1st, 1 is sys
            results = []
            for dialog in data:
                temp = []
                # convert utterance and feature into numeric numbers
                for usr_sent, sys_sent in dialog:
                    # if feat is not None:
                    #     id_feat = list(feat)
                    #     id_feat[self.dialog_act_id] = self.rev_dialog_act_vocab[feat[self.dialog_act_id]]
                    # else:
                    #     id_feat = None
                    temp_turn = [[self.rev_vocab.get(t, self.unk_id) for t in usr_sent],
                                 [self.rev_vocab.get(t, self.unk_id) for t in sys_sent]]
                    temp.append(temp_turn)
                results.append(temp)
            return results

        id_train = _to_id_corpus(self.train_corpus[self.dialog_id])
        if self.labeled:
            id_labeled = _to_id_corpus(self.labeled_corpus[self.dialog_id])
        id_test = _to_id_corpus(self.test_corpus[self.dialog_id])

        if self.labeled:
            return {'train': id_train, 'labeled': id_labeled, 'test': id_test}
        else:
            return {'train': id_train, 'test': id_test}

    def get_state_corpus(self, pad_to_length):
        def _one_hot_encoding(data):
            results = []
            for dialog in data:
                state_this_dialog = []
                for s in dialog:
                    enc = [0] * n_state
                    enc[int(s)] = 1
                    state_this_dialog.append(enc)
                for _ in xrange(pad_to_length-len(dialog)): # padding
                    state_this_dialog.append([0]*n_state)
                results.append(state_this_dialog)
            return results

        def _to_label_corpus(data):
            # enc = OneHoeEncoder(n_state)
            results = []
            for dialog in data:
                state_this_dialog = []
                for s in dialog:
                    # enc = [0] * n_state
                    # enc[int(s)] = 1
                    state_this_dialog.append(int(s))
                for _ in xrange(pad_to_length-len(dialog)): # padding
                    state_this_dialog.append(0)
                results.append(state_this_dialog[:pad_to_length])
            return results

        if self.labeled:
            id_labeled = _to_label_corpus(self.labeled_corpus[self.label_id])

        return {'labeled': id_labeled}

    def get_meta_corpus(self):
        def _to_id_corpus(data):
            results = []
            for m_meta, o_meta, topic in data:
                results.append((m_meta, o_meta, self.rev_topic_vocab[topic]))
            return results

        id_train = _to_id_corpus(self.train_corpus[self.meta_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.meta_id])
        id_test = _to_id_corpus(self.test_corpus[self.meta_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

