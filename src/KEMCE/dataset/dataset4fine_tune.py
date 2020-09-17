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


def collate_ml(batch, ent_embedding):

    input_lens = [len(x[0][1]) for x in batch]
    max_x_len = max(input_lens)

    input_mask_pad = []
    for l in input_lens:
        mask = np.zeros(max_x_len, dtype=np.float32)
        mask[:l] = 1
        input_mask_pad.append(mask)
    input_mask = np.stack(input_mask_pad)

    # padding input id
    input_pad = [pad1d(x[0][1], max_x_len) for x in batch]
    input = np.stack(input_pad)

    # padding entity id
    ent_input_pad = [pad1d(x[1][1], max_x_len) for x in batch]
    ent_input = np.stack(ent_input_pad)

    # generating entity mask
    ent_mask = np.array(ent_input > 0, dtype=np.float32)
    ent_mask[:, 0] = 1.

    # padding 2D description ID
    desc_input_pad = [pad2d(x[2][1], max_x_len) for x in batch]
    desc_input = np.stack(desc_input_pad)

    mortality_label_ls = [x[3][0] for x in batch]
    mortality_label = np.stack(mortality_label_ls)

    los_label_ls = [x[3][1] for x in batch]
    los_label = np.stack(los_label_ls)

    # embedding entity ids
    ent_input_tensor = torch.tensor(ent_input).long()
    ent_input_embed = ent_embedding(ent_input_tensor)

    output = {"input": torch.tensor(input).long(),
              "input_ent": ent_input_embed,
              "input_desc": torch.tensor(desc_input).long(),
              "mask_input": torch.tensor(input_mask),
              "mask_ent": torch.tensor(ent_mask),
              "label_los": torch.tensor(los_label),
              "label_mortality": torch.tensor(mortality_label)
              }
    return output


def collate_rd(batch, ent_embedding, num_dx_classes):

    input_lens = [len(x[0][1]) for x in batch]
    max_x_len = max(input_lens)

    input_mask_pad = []
    for l in input_lens:
        mask = np.zeros(max_x_len, dtype=np.float32)
        mask[:l] = 1
        input_mask_pad.append(mask)
    input_mask = np.stack(input_mask_pad)

    token_type_pad = []
    for x in batch:
        tokens = ' '.join(x[0][0])
        seps = tokens.split('[SEP]')
        sep_types = []
        for i, sep in enumerate(seps):
            if sep != '':
                length = len(sep.split())
                sep_types = sep_types + [i] * (length + 1)
        if len(sep_types) < max_x_len:
            sep_types = sep_types + [sep_types[-1]] * (max_x_len - len(sep_types))
        token_type_pad.append(np.array(sep_types))
    token_type = np.stack(token_type_pad)

    # padding input id
    input_pad = [pad1d(x[0][1], max_x_len) for x in batch]
    input = np.stack(input_pad)

    # padding entity id
    ent_input_pad = [pad1d(x[1][1], max_x_len) for x in batch]
    ent_input = np.stack(ent_input_pad)

    # generating entity mask
    ent_mask = np.array(ent_input > 0, dtype=np.float32)
    ent_mask[:, 0] = 1.

    # padding 2D description ID
    desc_input_pad = [pad2d(x[2][1], max_x_len) for x in batch]
    desc_input = np.stack(desc_input_pad)

    readmission_label_ls = [x[3][0] for x in batch]
    readmission_label = np.stack(readmission_label_ls)

    dx_classes = []
    for x in batch:
        dx_label = x[3][2]
        labels = np.zeros(num_dx_classes, dtype=np.float32)
        labels[dx_label] = 1.0
        dx_classes.append(labels)
    labels_dx = np.stack(dx_classes)

    # embedding entity ids
    ent_input_tensor = torch.tensor(ent_input).long()
    ent_input_embed = ent_embedding(ent_input_tensor)

    output = {"input": torch.tensor(input).long(),
              "input_ent": ent_input_embed,
              "input_desc": torch.tensor(desc_input).long(),
              "type_token": torch.tensor(token_type).long(),
              "mask_input": torch.tensor(input_mask),
              "mask_ent": torch.tensor(ent_mask),
              "label_dx": torch.tensor(labels_dx),
              "label_readm": torch.tensor(readmission_label)
              }
    return output


def load_data(sequences, labels):
    dataSize = len(labels)
    ind = np.random.permutation(dataSize)
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)

    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest + nValid]
    train_indices = ind[nTest + nValid:]

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
    def __init__(self, inputs, labels, seqs_tokenizer, ent_tokenize, desc_tokenize):
        self.inputs = inputs
        self.labels = labels
        self.seqs_tokenizer = seqs_tokenizer
        self.ent_tokenize = ent_tokenize
        self.desc_tokenize = desc_tokenize

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):

        input_str = self.inputs[item]
        ent_input_str = input_str.replace('[CLS]', '[UNK]').replace('[SEP]', '[UNK]')
        input_visit = self.seqs_tokenizer.tokenize(input_str)
        input_ent = self.ent_tokenize.tokenize(ent_input_str)
        input_desc = self.desc_tokenize.tokenize(ent_input_str)

        # label = self.labels[item]

        return input_visit, input_ent, input_desc, self.labels[item]


# if __name__ == '__main__':
#     dir_path = '../../../'
#
#     seqs_file = dir_path + 'outputs/kemce/data/raw/mimic_pre_train.seqs'
#     dict_file = dir_path + 'outputs/kemce/data/raw/mimic_pre_train_vocab.txt'
#     out_dir = dir_path + 'outputs/kemce/data/raw/'
#     tokenizer = SeqsTokenizer(dict_file)
#
#     prepare_data(tokenizer.ids_to_tokens, seqs_file, out_dir)

