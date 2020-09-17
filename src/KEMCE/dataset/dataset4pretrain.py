from torch.utils.data import Dataset
import random
import torch
import pickle
import numpy as np
from KEMCE.knowledge_bert import SeqsTokenizer


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len):
    shape = x.shape
    pad = np.zeros((max_len-shape[0], shape[1]), dtype=np.long)
    return np.concatenate([x, pad])


def collate_mlm(batch):

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
        tokens = x[0][0]
        mask = np.ones(max_x_len, dtype=np.float32)
        for i, token in enumerate(tokens):
            if token != '[SEP]':
                mask[i] = 0
            elif token == '[SEP]':
                mask[i] = 0
                break
        token_type_pad.append(mask)
    token_type = np.stack(token_type_pad)

    # padding input id
    mlm_input_pad = [pad1d(x[0][1], max_x_len) for x in batch]
    mlm_input = np.stack(mlm_input_pad)

    # padding label id
    mlm_label_pad = [pad1d(x[1][1], max_x_len) for x in batch]
    mlm_label = np.stack(mlm_label_pad)

    # padding entity id
    ent_input_pad = [pad1d(x[2][1], max_x_len) for x in batch]
    ent_input = np.stack(ent_input_pad)

    # generating entity mask
    ent_mask = np.array(ent_input > 0, dtype=np.float32)
    ent_mask[:, 0] = 1.

    # # padding 2D description ID
    # desc_input_pad = [pad2d(x[3][1], max_x_len) for x in batch]
    # desc_input = np.stack(desc_input_pad)

    next_sent_label_ls = [x[3] for x in batch]
    next_sent_label = np.stack(next_sent_label_ls)

    # embedding entity ids
    # ent_input_tensor = torch.tensor(ent_input).long()
    # ent_input_embed = ent_embedding(ent_input_tensor)

    output = {"mlm_input": torch.tensor(mlm_input).long(),
              "mlm_label": torch.tensor(mlm_label).long(),
              "ent_input": torch.tensor(ent_input).long(),
              # "desc_input": torch.tensor(desc_input).long(),
              "input_mask": torch.tensor(input_mask),
              "ent_mask": torch.tensor(ent_mask),
              "token_type": torch.tensor(token_type).long(),
              "next_sent": torch.tensor(next_sent_label)
              }

    return output


class BERTDataset(Dataset):
    def __init__(self, input_pairs, input_mask_pairs, input_pair_mask_labels,
                 seqs_tokenizer, ent_tokenize, desc_tokenize=None):
        self.input_pairs = input_pairs
        self.input_mask_pairs = input_mask_pairs
        self.input_pair_mask_labels = input_pair_mask_labels
        self.seqs_tokenizer = seqs_tokenizer
        self.ent_tokenize = ent_tokenize
        # self.desc_tokenize = desc_tokenize

    def __len__(self):
        return len(self.input_pairs)

    def __getitem__(self, item):

        pairs, _ = self.input_pairs[item]
        mask_pairs, type_label = self.input_mask_pairs[item]
        mask_label = self.input_pair_mask_labels[item]

        ent_input_ls = ['[UNK]'] + list(pairs[0]) + ['[UNK]'] + list(pairs[1]) + ['[UNK]']
        ent_input_str = ' '.join(ent_input_ls)
        mlm_input_ls = ['[CLS]'] + mask_pairs[0] + ['[SEP]'] + mask_pairs[1] + ['[SEP]']
        mlm_input_str = ' '.join(mlm_input_ls)
        mlm_label_ls = ['[PAD]'] + mask_label[0] + ['[PAD]'] + mask_label[1] + ['[PAD]']
        mlm_label_str = ' '.join(mlm_label_ls)

        masked_input = self.seqs_tokenizer.tokenize(mlm_input_str)
        masked_label = self.seqs_tokenizer.tokenize(mlm_label_str)
        ent_input = self.ent_tokenize.tokenize(ent_input_str)
        # desc_input = self.desc_tokenize.tokenize(ent_input_str)
        next_sentence_label = type_label

        # return masked_input, masked_label, ent_input, desc_input, next_sentence_label
        return masked_input, masked_label, ent_input, next_sentence_label


def prepare_data(vocab, seqs_file, out_file):
    # format of vocab is index:code
    # seqs_file = 'outputs/kemce/data/raw/mimic_pre_train.seqs'
    # dict_file = 'outputs/kemce/data/raw/mimic_pre_train_vocab.txt'
    patient_seqs = pickle.load(open(seqs_file, 'rb'))

    # vocab = load_vocab_seqs(dict_file)
    seqs_one = []
    seqs_two_plus = []
    for seq in patient_seqs:
        if len(seq) > 1:
            seqs_two_plus.append(seq)
        else:
            seqs_one.append(seq)

    true_pair_seqs = []
    for seq in seqs_two_plus:
        length = len(seq)
        for i in range(length - 1):
            true_pair_seqs.append([seq[i], seq[i + 1]])

    false_pair_seqs = []
    length = len(seqs_one)
    while length > 1:
        sampling = random.sample(range(length), k=2)
        sampling = sorted(sampling, reverse=True)
        seq1 = seqs_one.pop(sampling[0])
        seq2 = seqs_one.pop(sampling[1])
        false_pair_seqs.append([seq1[0], seq2[0]])
        length = len(seqs_one)
    if len(seqs_one) > 0:
        false_pair_seqs.append([seqs_one.pop()[0], false_pair_seqs[0][0]])

    pair_seqs = []
    pair_seqs_mask = []
    pair_seqs_mask_label = []

    for i, seq_pairs in enumerate([true_pair_seqs,false_pair_seqs]):
        for seq_pair in seq_pairs:
            pairs = []
            pairs_label = []
            for seq in seq_pair:
                seq_mask = []
                seq_mask_label = []
                for code in seq:
                    prob = random.random()
                    if prob < 0.15:
                        prob /= 0.15
                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            seq_mask.append('[MASK]')
                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            seq_mask.append(vocab[random.randrange(len(vocab))])
                        # 10% randomly change token to current token
                        else:
                            seq_mask.append(code)
                        seq_mask_label.append(code)
                    else:
                        seq_mask.append(code)
                        seq_mask_label.append('[PAD]')
                pairs.append(seq_mask)
                pairs_label.append(seq_mask_label)
            pair_seqs.append((seq_pair,i))
            pair_seqs_mask.append((pairs, i))
            pair_seqs_mask_label.append(pairs_label)

    pickle.dump(pair_seqs, open(out_file + 'mimic_pre_train_input_pairs.pickle', 'wb'), -1)
    pickle.dump(pair_seqs_mask, open(out_file + 'mimic_pre_train_input_pairs_mask.pickle', 'wb'), -1)
    pickle.dump(pair_seqs_mask_label, open(out_file + 'mimic_pre_train_input_pairs_mask_labels.pickle', 'wb'), -1)

    return pair_seqs, pair_seqs_mask, pair_seqs_mask_label


if __name__ == '__main__':
    dir_path = '../../../'

    seqs_file = dir_path + 'outputs/kemce/data/raw/mimic_pre_train.seqs'
    dict_file = dir_path + 'outputs/kemce/data/raw/mimic_pre_train_vocab.txt'
    out_dir = dir_path + 'outputs/kemce/data/raw/'
    tokenizer = SeqsTokenizer(dict_file)

    prepare_data(tokenizer.ids_to_tokens, seqs_file, out_dir)

