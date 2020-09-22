from torch.utils.data import Dataset
import torch
import numpy as np
from KAME.kame_helpers import codes2ancestors
_TEST_RATIO = 0.15
_VALIDATION_RATIO = 0.1


def collate(batch, num_class, num_leaf_code):

    lengths = np.array([len(x[0]) for x in batch]) - 1
    n_samples = len(batch)
    maxlen = np.max(lengths)

    lengths_ans = []
    for x in batch:
        seq = x[1]
        for visit in seq:
            lengths_ans.append(len(visit))
    maxlen_ans = np.max(np.array(lengths_ans))

    x = np.zeros((n_samples, maxlen, num_leaf_code)).astype(np.float32)
    f = np.zeros((n_samples, maxlen, maxlen_ans)).astype(np.long)
    y = np.zeros((n_samples, maxlen, num_class)).astype(np.float32)
    last_y = np.zeros((n_samples, num_class)).astype(np.float32)
    mask = np.zeros((n_samples, maxlen)).astype(np.float32)

    for idx, (seq, ans_seq, lseq) in enumerate(batch):
        for xvec, subseq in zip(x[idx,:,:], seq[:-1]):
            xvec[subseq] = 1.
        for fvec, subseq in zip(f[idx,:,:], ans_seq[:-1]):
            fvec[:len(subseq)] = subseq
        for yvec, subseq in zip(y[idx, :, :], lseq[1:]):
            yvec[subseq] = 1.
        mask[idx, :lengths[idx]] = 1.
        last_y[idx] = y[idx, lengths[idx] - 1, :]

    lengths = np.array(lengths, dtype=np.float32)

    output = {"input": torch.tensor(x),
              "input_ans": torch.tensor(f),
              "label_dx": torch.tensor(y),
              'label_last': torch.tensor(last_y),
              "mask": torch.tensor(mask),
              "lengths": torch.tensor(lengths)
              }
    return output


def load_data(sequences, labels, leaf2ans, num_leaves):
    dataSize = len(labels)
    ind = np.random.permutation(dataSize)
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)

    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest+nValid]
    train_indices = ind[nTest+nValid:]

    train_set_x = [sequences[i] for i in train_indices]
    train_set_y = [labels[i] for i in train_indices]
    test_set_x = [sequences[i] for i in test_indices]
    test_set_y = [labels[i] for i in test_indices]
    valid_set_x = [sequences[i] for i in valid_indices]
    valid_set_y = [labels[i] for i in valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_f = [codes2ancestors(train_set_x[i], leaf2ans, num_leaves) for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_f = [codes2ancestors(valid_set_x[i], leaf2ans, num_leaves) for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_f = [codes2ancestors(test_set_x[i], leaf2ans, num_leaves) for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_f, train_set_y)
    valid_set = (valid_set_x, valid_set_f, valid_set_y)
    test_set = (test_set_x, test_set_f, test_set_y)

    return train_set, valid_set, test_set


class FTDataset(Dataset):
    def __init__(self, inputs, ans_seqs, labels):
        self.inputs = inputs
        self.labels = labels
        self.ans_seqs = ans_seqs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item], self.ans_seqs[item], self.labels[item]


