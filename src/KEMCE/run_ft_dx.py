from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
import datetime
import time
import copy
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import numpy as np
from shutil import copyfile
import torch
import pickle
from KEMCE.knowledge_bert import SeqsTokenizer, EntityTokenizer, DescTokenizer, \
    KemceDxPrediction, BertAdam, BertConfig
from KEMCE.dataset import FTDataset, collate_rd, load_data
from GRAM.gram_helpers import print2file
from KEMCE.utils.evaluation import PredictionEvaluation as Evaluation


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

    ## Required parameters
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

    ## Other parameters
    parser.add_argument("--pretrained_dir",
                        default=None,
                        type=str,
                        # required=True,
                        help="The pre_trained model directory.")
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
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

    output_dir = os.path.join(args.output_dir, args.task)
    log_file = os.path.join(output_dir, 'dx_prediction.log')
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    data_path = args.data_dir

    # training data files
    seqs_file =   data_path + 'outputs/kemce/data/readmission_diagnosis/mimic.inputs_all.seqs'
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

    # load data
    data_dict = dict()
    train_set, valid_set, test_set = load_data(inputs, labels)

    train_dataset = FTDataset(train_set[0], train_set[1], seq_tokenizer, ent_tokenize, desc_tokenize)
    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                   collate_fn=lambda batch: collate_rd(batch, ent_embd, config.num_ccs_classes),
                                   num_workers=0, shuffle=True)
    size_train_data = len(train_set[0])
    num_train_steps = int(size_train_data / args.train_batch_size * args.num_train_epochs)

    val_dataset = FTDataset(valid_set[0], valid_set[1], seq_tokenizer, ent_tokenize, desc_tokenize)
    val_data_loader = DataLoader(val_dataset, batch_size=args.train_batch_size,
                                   collate_fn=lambda batch: collate_rd(batch, ent_embd, config.num_ccs_classes),
                                   num_workers=0, shuffle=True)
    size_val_data = len(valid_set[0])
    num_val_steps = int(size_val_data / args.train_batch_size * args.num_train_epochs)

    test_dataset = FTDataset(test_set[0], test_set[1], seq_tokenizer, ent_tokenize, desc_tokenize)
    test_data_loader = DataLoader(test_dataset, batch_size=args.train_batch_size,
                                 collate_fn=lambda batch: collate_rd(batch, ent_embd, config.num_ccs_classes),
                                 num_workers=0, shuffle=True)
    size_test_data = len(test_set[0])
    num_test_steps = int(size_test_data / args.train_batch_size * args.num_train_epochs)

    data_dict['train'] = [train_data_loader, size_train_data, num_train_steps]
    data_dict['val'] = [val_data_loader, size_val_data, num_val_steps]
    data_dict['test'] = [test_data_loader, size_test_data, num_test_steps]

    # data_file = os.path.join(output_dir, 'mimic_' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '.data')
    # # prepared_data_file = data_file + '_' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '.data'
    # print2file('prepared data saved to: {}'.format(data_file), log_file)
    # pickle.dump(data_dict, open(data_file, 'wb'), -1)

    log_file = os.path.join(output_dir, 'dx_prediction.log')

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
    # if args.do_train:


    # model.train()
    fout = open(os.path.join(output_dir, "loss.{}".format(datetime.datetime.now())), 'w')
    best_accuracy_at_top_5 = 0
    epoch_duration = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:

            buf = '{} ********** Running {} ***********'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                                   phase)
            print2file(buf, log_file)
            buf = '{} Num examples = {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                                data_dict[phase][1])
            print2file(buf, log_file)
            buf = '{}  Batch size = {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                args.train_batch_size)
            print2file(buf, log_file)
            buf = '{}  Num steps = {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                               data_dict[phase][2])
            print2file(buf, log_file)

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            data_iter = iter(data_dict[phase][0])
            tr_loss = 0
            accuracy_ls = []
            start_time = time.time()
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(data_iter, desc="Iteration")):
                batch = {k: t.to(device) for k, t in batch.items()}

                input_ids = batch['input']
                type_token = batch['type_token']
                input_ent = batch['input_ent']
                input_desc = batch['input_desc']
                ent_mask = batch['mask_ent']
                input_mask = batch['mask_input']
                label_task = batch['label_dx']

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = model(input_ids,
                                     type_token,
                                     input_ent,
                                     input_desc,
                                     ent_mask,
                                     input_mask,
                                     label_task)
                        loss.backward()
                        optimizer.step()
                        lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                        optimizer.zero_grad()
                        global_step += 1

                        fout.write("{}\n".format(loss.item()))
                        tr_loss += loss.item()
                        nb_tr_examples += input_ids.size(0)
                        nb_tr_steps += 1
                    else:
                        outputs = model( input_ids,
                                         type_token,
                                         input_ent,
                                         input_desc,
                                         ent_mask,
                                         input_mask)
                        predicts = outputs.cpu().detach().numpy()
                        trues = label_task.cpu().numpy()
                        predicts = predicts.reshape(-1, predicts.shape[-1])
                        trues = trues.reshape(-1, trues.shape[-1])

                        # recalls = Evaluation.visit_level_precision_at_k(trues, predicts)
                        accuracy = Evaluation.code_level_accuracy_at_k(trues, predicts)
                        # precision_lst.append(recalls)
                        accuracy_ls.append(accuracy)

            duration = time.time() - start_time
            if phase == 'train':
                fout.write("train loss {} on epoch {}\n".format(epoch, tr_loss/nb_tr_steps))
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(output_dir, "pytorch_model.bin_{}".format(epoch))
                torch.save(model_to_save.state_dict(), output_model_file)
                buf = '{} {} Loss: {:.4f}, Duration: {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), phase, tr_loss/nb_tr_steps, duration)
                print2file(buf, log_file)
                epoch_duration += duration
            else:
                # epoch_precision = (np.array(precision_lst)).mean(axis=0)
                epoch_accuracy = (np.array(accuracy_ls)).mean(axis=0)
                buf = '{} {} Accuracy: {}, Duration: {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), phase, epoch_accuracy, duration)
                print2file(buf, log_file)
                if phase == 'val' and epoch_accuracy[0] > best_accuracy_at_top_5:
                    best_accuracy_at_top_5 = epoch_accuracy[0]
                    best_model_wts = copy.deepcopy(model.state_dict())
    fout.close()

    buf = '{} Training complete in {:.0f}m {:.0f}s'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                           epoch_duration // 60, epoch_duration % 60)
    print2file(buf, log_file)

    buf = '{} Best accuracy at top 5: {:4f}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                    best_accuracy_at_top_5)
    print2file(buf, log_file)

    # load best model weights
    model.load_state_dict(best_model_wts)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    buf = '{} Save the best model to {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), output_model_file)
    print2file(buf, log_file)
    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)

    # Save the optimizer
    output_optimizer_file = os.path.join(output_dir, "pytorch_op.bin")
    torch.save(optimizer.state_dict(), output_optimizer_file)


if __name__ == "__main__":
    main()


