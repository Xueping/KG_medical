from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import numpy as np
from shutil import copyfile
import torch
import pickle
from KEMCE.knowledge_bert import SeqsTokenizer, EntityTokenizer, DescTokenizer, \
    KemceDxPrediction, BertAdam, BertConfig
from KEMCE.dataset import FTDataset, collate_rd


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task",
                        default=None,
                        type=str,
                        required=True,
                        help="The prediction task, such as Mortality (mort) or Length of stay (los).")

    # Other parameters
    parser.add_argument("--pretrained_dir",
                        default=None,
                        type=str,
                        # required=True,
                        help="The pre_trained model directory.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--gpu",
                        default=0,
                        type=int,
                        help="CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    output_dir = os.path.join(args.output_dir, args.task)
    data_path = args.data_dir

    # training data files
    seqs_file = data_path + 'outputs/kemce/data/readmission_diagnosis/mimic.inputs_all.seqs'
    labels_file = data_path + 'outputs/kemce/data/readmission_diagnosis/mimic.labels_all.label'

    # dictionary files
    dict_file = data_path + 'outputs/kemce/data/raw/mimic_pre_train_vocab.txt'
    code2desc_file = data_path + 'outputs/kemce/KG/code2desc.pickle'
    ent_vocab_file = data_path + 'outputs/kemce/KG/entity2id'

    # entity embedding file
    ent_embd_file = data_path + 'outputs/kemce/KG/embeddings/CCS_TransR_entity.npy'

    # model configure file
    config_json = data_path + 'src/KEMCE/kemce_config.json'
    copyfile(config_json, os.path.join(output_dir, 'config.json'))

    inputs = pickle.load(open(seqs_file, 'rb'))
    labels = pickle.load(open(labels_file, 'rb'))

    # loading entity embedding matrix
    ent_embd = np.load(ent_embd_file)
    ent_embd = torch.tensor(ent_embd)
    # padding for special word "unknown"
    pad_embed = torch.zeros(1, ent_embd.shape[1])
    ent_embd = torch.cat([pad_embed, ent_embd])
    ent_embd = torch.nn.Embedding.from_pretrained(ent_embd, freeze=True)
    logger.info("Shape of entity embedding: " + str(ent_embd.weight.size()))

    # initialize tokenizers
    seq_tokenizer = SeqsTokenizer(dict_file)
    ent_tokenize = EntityTokenizer(ent_vocab_file)
    desc_tokenize = DescTokenizer(code2desc_file)

    # load configure file
    config = BertConfig.from_json_file(config_json)

    # save vocabulary to output directory
    vocab_out = open(os.path.join(output_dir, "mimic_kemce_vocab.txt"), 'w')
    vocab = seq_tokenizer.ids_to_tokens
    size_vocab = len(vocab)
    for i in range(size_vocab):
        vocab_out.write(vocab[i]+'\n')
    vocab_out.close()

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info("device: {}".format(device))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    size_train_data = None
    num_train_steps = None
    train_data_loader = None
    if args.do_train:
        dataset = FTDataset(inputs, labels, seq_tokenizer, ent_tokenize, desc_tokenize)
        train_data_loader = DataLoader(dataset, batch_size=args.train_batch_size,
                                       collate_fn=lambda batch: collate_rd(batch, ent_embd, config.num_ccs_classes),
                                       num_workers=0, shuffle=True)
        size_train_data = len(inputs)
        num_train_steps = int(size_train_data / args.train_batch_size * args.num_train_epochs)

    if args.pretrained_dir is not None:
        weights_path = os.path.join(args.pretrained_dir, 'pytorch_model.bin_56')
        state_dict = torch.load(weights_path)
        model, _ = KemceDxPrediction.from_pretrained(args.pretrained_dir, state_dict)
    else:
        model = KemceDxPrediction(config)
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_linear = ['layer.11.output.dense_ent',
                 'layer.11.output.LayerNorm_ent']
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nl in n for nl in no_linear)]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight',
                'LayerNorm_ent.bias', 'LayerNorm_ent.weight',
                'LayerNorm_desc.bias', 'LayerNorm_desc.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    t_total = num_train_steps

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", size_train_data)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        model.train()
        import datetime
        fout = open(os.path.join(output_dir, "loss.{}".format(datetime.datetime.now())), 'w')
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            dataiter = iter(train_data_loader)
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(dataiter, desc="Iteration")):
                batch = {k: t.to(device) for k, t in batch.items()}

                input_ids = batch['input']
                type_token = batch['type_token']
                input_ent = batch['input_ent']
                input_desc = batch['input_desc']
                ent_mask = batch['mask_ent']
                input_mask = batch['mask_input']
                label_task = batch['label_dx']

                loss = model(input_ids,
                             type_token,
                             input_ent,
                             input_desc,
                             ent_mask,
                             input_mask,
                             label_task)

                loss.backward()

                fout.write("{}\n".format(loss.item()))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            fout.write("utils loss {} on epoch {}\n".format(epoch, tr_loss/nb_tr_steps))

        fout.close()



if __name__ == "__main__":
    main()


