#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
from __future__ import print_function
from __future__ import division

import numpy as np


# Data feed
class LongDataLoader(object):
    """A special efficient data loader for TBPTT"""
    batch_size = 0
    backward_size = 0
    step_size = 0
    ptr = 0
    num_batch = None
    batch_indexes = None
    grid_indexes = None
    indexes = None
    data_lens = None
    data_size = None
    prev_alive_size = 0
    name = None

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, cur_grid, prev_grid):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, shuffle=True, intra_shuffle=True):
        assert len(self.indexes) == self.data_size and len(self.data_lens) == self.data_size

        self.ptr = 0
        self.batch_size = batch_size
        self.prev_alive_size = batch_size

        # create batch indexes
        temp_num_batch = self.data_size // batch_size
        self.batch_indexes = []
        for i in range(temp_num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        left_over = self.data_size-temp_num_batch*batch_size

        # shuffle batch indexes
        if shuffle:
            self._shuffle_batch_indexes()

        """
        # create grid indexes
        self.grid_indexes = []
        for idx, b_ids in enumerate(self.batch_indexes):
            # assume the b_ids are sorted
            all_lens = [self.data_lens[i] for i in b_ids]
            max_len = self.data_lens[b_ids[-1]]
            min_len = self.data_lens[b_ids[0]]
            assert np.max(all_lens) == max_len
            assert np.min(all_lens) == min_len
            num_seg = (max_len-self.backward_size) // self.step_size
            if num_seg > 0:
                cut_start = range(0, num_seg*self.step_size, step_size)
                cut_end = range(self.backward_size, num_seg*self.step_size+self.backward_size, step_size)
                assert cut_end[-1] < max_len
                cut_start = [0] * (self.backward_size-2) +cut_start # since we give up on the seq training idea
                cut_end = range(2, self.backward_size) + cut_end
            else:
                cut_start = [0] * (max_len-2)
                cut_end = range(2, max_len)

            new_grids = [(idx, s_id, e_id) for s_id, e_id in zip(cut_start, cut_end) if s_id < min_len-1]
            if intra_shuffle and shuffle:
               np.random.shuffle(new_grids)
            self.grid_indexes.extend(new_grids)
        """

        self.num_batch = len(self.batch_indexes)
        print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_index_list = self.batch_indexes[self.ptr]
            # if self.ptr > 0:
            #     prev_grid = self.grid_indexes[self.ptr-1]
            # else:
            #     prev_grid = None
            self.ptr += 1
            return self._prepare_batch(current_index_list)
        else:
            if self.labeled:
                current_index_list = self.batch_indexes[0]
                return self._prepare_batch(current_index_list)
            else:
                return None


class SWDADataLoader(LongDataLoader):
    def __init__(self, name, data, config, labeled=False):
        # assert len(data) == len(meta_data)
        self.name = name
        self.data = data
        # self.meta_data = meta_data
        self.data_size = len(data)
        self.data_lens = all_lens = [len(line) for line in self.data]
        self.max_utt_size = config.max_utt_len
        self.max_dialog_size = config.max_dialog_len
        self.labeled = labeled
        print("Max dialog len %d and min dialog len %d and avg len %f" % (np.max(all_lens),
                                                            np.min(all_lens),
                                                            float(np.mean(all_lens))))
        # self.indexes = list(np.argsort(all_lens))
        self.indexes = range(self.data_size)
        np.random.shuffle(self.indexes)

    def pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_utt_size:
            return tokens[0:(self.max_utt_size-1)] + [tokens[-1]], [1] * self.max_utt_size
        elif do_pad:
            return tokens + [0] * (self.max_utt_size-len(tokens)), [1] * len(tokens) + [0] * (self.max_utt_size-len(tokens))
        else:
            return tokens

    def pad_dialog(self, dialog):
        dialog_usr_input, dialog_sys_input, dialog_usr_mask, dialog_sys_mask = [], [], [], []
        if len(dialog) >= self.max_dialog_size:
            for turn in dialog[:self.max_dialog_size]:
                usr_input, usr_mask = self.pad_to(turn[0])
                sys_input, sys_mask = self.pad_to(turn[1])
                dialog_usr_input.append(usr_input)
                dialog_sys_input.append(sys_input)
                dialog_usr_mask.append(usr_mask)
                dialog_sys_mask.append(sys_mask)
        else:
            all_pad_input, all_pad_mask = self.pad_to([])
            for turn in dialog:
                usr_input, usr_mask = self.pad_to(turn[0])
                sys_input, sys_mask = self.pad_to(turn[1])
                dialog_usr_input.append(usr_input)
                dialog_sys_input.append(sys_input)
                dialog_usr_mask.append(usr_mask)
                dialog_sys_mask.append(sys_mask)
            for _ in range(self.max_dialog_size-len(dialog)):
                dialog_usr_input.append(all_pad_input)
                dialog_sys_input.append(all_pad_input)
                dialog_usr_mask.append(all_pad_mask)
                dialog_sys_mask.append(all_pad_mask)
        assert len(dialog_usr_input) == len(dialog_sys_input) == len(dialog_usr_mask) == len(dialog_sys_mask) == self.max_dialog_size
        return dialog_usr_input, dialog_sys_input, dialog_usr_mask, dialog_sys_mask


    def _prepare_batch(self, cur_index_list):
        # the batch index, the starting point and end point for segment
        # need usr_input_sent, sys_input_sent, dialog_len_mask, usr_full_mask, sys_full_mask = batch

        dialogs = [self.data[idx] for idx in cur_index_list]
        dialog_lens = [self.data_lens[idx] for idx in cur_index_list]

        usr_input_sent, sys_input_sent, usr_full_mask, sys_full_mask = [], [], [], []
        for dialog in dialogs:
            dialog_usr_input, dialog_sys_input, dialog_usr_mask, dialog_sys_mask = self.pad_dialog(dialog)
            usr_input_sent.append(dialog_usr_input)
            sys_input_sent.append(dialog_sys_input)
            usr_full_mask.append(dialog_usr_mask)
            sys_full_mask.append(dialog_sys_mask)

        # initial_prev_zt = np.ones()

        return np.array(usr_input_sent), np.array(sys_input_sent), np.array(dialog_lens), \
               np.array(usr_full_mask), np.array(sys_full_mask)





