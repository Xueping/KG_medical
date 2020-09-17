from torch.utils.data import Dataset
import torch
import numpy as np
_TEST_RATIO = 0.15
_VALIDATION_RATIO = 0.1


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len):
    shape = x.shape
    pad = np.zeros((max_len-shape[0], shape[1]), dtype=np.long)
    return np.concatenate([x, pad])


def collate_rd(batch, num_dx_classes):

    seq_lens = [len(x[0]) for x in batch]
    max_seq_len = max(seq_lens)

    visit_mask_pad = []
    for l in seq_lens:
        mask = np.zeros(max_seq_len, dtype=np.float32)
        mask[:l] = 1
        visit_mask_pad.append(mask)
    visit_mask = np.stack(visit_mask_pad)

    max_visit_len = 0
    for x in batch:
        for visit in x[0]:
            if max_visit_len < len(visit):
                max_visit_len = len(visit)

    inputs_pad = []
    for x in batch:
        seq_len = len(x[0])
        visit_pad = np.zeros((max_seq_len-seq_len, max_visit_len), dtype=np.float32)
        visit_codes = []
        for visit in x[0]:
            visit_len = len(visit)
            code_pad = visit + [0]*(max_visit_len-visit_len)
            visit_codes.append(code_pad)
        seq_pad = np.concatenate([np.stack(visit_codes), visit_pad], axis=0)
        inputs_pad.append(seq_pad)
    inputs = np.stack(inputs_pad)

    code_mask = np.array(inputs > 0, dtype=np.float32)

    readmission_label_ls = [x[1][0] for x in batch]
    readmission_label = np.stack(readmission_label_ls)

    dx_classes = []
    for x in batch:
        dx_label = x[1][2]
        labels = np.zeros(num_dx_classes, dtype=np.float32)
        labels[dx_label] = 1.0
        dx_classes.append(labels)
    labels_dx = np.stack(dx_classes)

    output = {"input": torch.tensor(inputs).long(),
              "visit_mask": torch.tensor(visit_mask),
              "code_mask": torch.tensor(code_mask),
              "label_dx": torch.tensor(labels_dx),
              "label_readm": torch.tensor(readmission_label)
              }
    return output


def load_data(sequences, labels):

    data_size = len(labels)
    ind = np.random.permutation(data_size)
    n_test = int(_TEST_RATIO * data_size)
    n_valid = int(_VALIDATION_RATIO * data_size)

    test_indices = ind[:n_test]
    valid_indices = ind[n_test:n_test + n_valid]
    train_indices = ind[n_test + n_valid:]

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
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)

    return train_set, valid_set, test_set


class FTDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]


# if __name__ == '__main__':
#     dir_path = '../../../'
#
#     seqs_file = dir_path + 'outputs/kemce/data/raw/mimic_pre_train.seqs'
#     dict_file = dir_path + 'outputs/kemce/data/raw/mimic_pre_train_vocab.txt'
#     out_dir = dir_path + 'outputs/kemce/data/raw/'
#     tokenizer = SeqsTokenizer(dict_file)
#
#     prepare_data(tokenizer.ids_to_tokens, seqs_file, out_dir)

