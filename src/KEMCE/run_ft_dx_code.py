from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('/home/xpeng/research/projects/medicalAI_torch/src/KG_medical/src')
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
import torch.optim as optim
import pickle
from KEMCE.dag_code import BertConfig, DiagnosisPrediction, FTDataset, collate_rd, load_data
from GRAM.gram_helpers import print2file
from KEMCE.utils.evaluation import PredictionEvaluation as Evaluation
from GRAM.gram_helpers import build_tree, get_rootCode

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
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
    parser.add_argument("--data_source",
                        default='mimic',
                        type=str,
                        # required=True,
                        help="the data source: mimic III (mimic) or eICU (eicu).")
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
    parser.add_argument("--add_dag",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lamda",
                        default=1.0,
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

    task = args.task + '_lr_' + str(args.learning_rate) + '_bs_' + str(args.train_batch_size)  + \
           '_e_' + str(args.num_train_epochs) + '_l_' + str(args.lamda)
    output_dir = os.path.join(args.output_dir, args.data_source, task)
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'dx_prediction.log')
    buf = '{} seed:{}, gpu:{}, num_train_epochs:{}, learning_rate:{}, train_batch_size:{}, output_dir:{}'.format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        args.seed, args.gpu, int(args.num_train_epochs), args.learning_rate, args.train_batch_size,
        output_dir)
    print2file(buf, log_file)

    data_path = args.data_dir
    if args.data_source == 'mimic':
        # training data files
        seqs_file = data_path + 'outputs/kemce/data/seq_prediction/mimic.seqs'
        labels_file = data_path + 'outputs/kemce/data/seq_prediction/mimic.labels_ccs.label'
        # labels_file = data_path + 'outputs/kemce/data/seq_prediction/mimic.labels_all.label'
        labels_visit_file = data_path + 'outputs/kemce/data/seq_prediction/mimic.labels_visit_cat1.label'
        # dictionary files
        dict_file = data_path + 'outputs/kemce/data/seq_prediction/mimic.types'
        tree_file = data_path + 'outputs/kemce/data/seq_prediction/mimic'
        # class_dict_file = data_path + 'outputs/kemce/data/seq_prediction/mimic.3digitICD9.dict'
        class_dict_file = data_path + 'outputs/kemce/data/seq_prediction/mimic.ccs_single_level.dict'
        visit_class_dict_file = data_path + 'outputs/kemce/data/seq_prediction/mimic.ccs_cat1.dict'
    else:
        # training data files
        seqs_file = data_path + 'outputs/eICU/seq_prediction/eicu.seqs'
        labels_file = data_path + 'outputs/eICU/seq_prediction/eicu.labels_ccs.label'
        labels_visit_file = data_path + 'outputs/eICU/seq_prediction/eicu.labels_visit_cat1.label'
        # dictionary files
        dict_file = data_path + 'outputs/eICU/seq_prediction/eicu.types'
        tree_file = data_path + 'outputs/eICU/seq_prediction/eicu'
        class_dict_file = data_path + 'outputs/eICU/seq_prediction/eicu.ccs_single_level.dict'
        visit_class_dict_file = data_path + 'outputs/eICU/seq_prediction/eicu.ccs_cat1.dict'

    # model configure file
    config_json = 'dag_code/config.json'
    copyfile(config_json, os.path.join(output_dir, 'config.json'))

    inputs = pickle.load(open(seqs_file, 'rb'))
    labels = pickle.load(open(labels_file, 'rb'))
    labels_visit = pickle.load(open(labels_visit_file, 'rb'))

    leaves_list = []
    ancestors_list = []
    for i in range(5, 0, -1):
        leaves, ancestors = build_tree(tree_file + '.level' + str(i) + '.pk')
        leaves_list.extend(leaves)
        ancestors_list.extend(ancestors)

    # load configure file
    config = BertConfig.from_json_file(config_json)

    config.leaves_list = leaves_list
    config.ancestors_list = ancestors_list
    vocab = pickle.load(open(dict_file, 'rb'))
    config.code_size = len(vocab)
    num_tree_nodes = get_rootCode(tree_file + '.level2.pk') + 1
    config.num_tree_nodes = num_tree_nodes
    class_vocab = pickle.load(open(class_dict_file, 'rb'))
    config.num_ccs_classes = len(class_vocab)
    visit_class_vocab = pickle.load(open(visit_class_dict_file, 'rb'))
    config.num_visit_classes = len(visit_class_vocab)

    config.add_dag = args.add_dag
    config.lamda = args.lamda

    max_seqs_len = 0
    for seq in inputs:
        if len(seq) > max_seqs_len:
            max_seqs_len = len(seq)
    config.max_position_embeddings = max_seqs_len

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info("device: {}".format(device))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data
    data_dict = dict()
    train_set, valid_set, test_set = load_data(inputs, labels, labels_visit)

    train_dataset = FTDataset(train_set[0], train_set[1], train_set[2])
    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                   collate_fn=lambda batch: collate_rd(batch, config.num_ccs_classes,
                                                                       config.num_visit_classes),
                                   num_workers=0, shuffle=True)
    size_train_data = len(train_set[0])
    num_train_steps = int(size_train_data / args.train_batch_size * args.num_train_epochs)

    val_dataset = FTDataset(valid_set[0], valid_set[1], valid_set[2])
    val_data_loader = DataLoader(val_dataset, batch_size=args.train_batch_size,
                                 collate_fn=lambda batch: collate_rd(batch, config.num_ccs_classes,
                                                                     config.num_visit_classes),
                                 num_workers=0, shuffle=True)
    size_val_data = len(valid_set[0])
    num_val_steps = int(size_val_data / args.train_batch_size * args.num_train_epochs)

    test_dataset = FTDataset(test_set[0], test_set[1], test_set[2])
    test_data_loader = DataLoader(test_dataset, batch_size=args.train_batch_size,
                                  collate_fn=lambda batch: collate_rd(batch, config.num_ccs_classes,
                                                                      config.num_visit_classes),
                                  num_workers=0, shuffle=True)
    size_test_data = len(test_set[0])
    num_test_steps = int(size_test_data / args.train_batch_size * args.num_train_epochs)

    data_dict['train'] = [train_data_loader, size_train_data, num_train_steps]
    data_dict['val'] = [val_data_loader, size_val_data, num_val_steps]
    data_dict['test'] = [test_data_loader, size_test_data, num_test_steps]

    model = DiagnosisPrediction(config)
    model.to(device)

    params_to_update = model.parameters()
    optimizer = optim.Adadelta(params_to_update, lr=args.learning_rate)
    # optimizer = optim.AdamW(params_to_update)

    fout = open(os.path.join(output_dir, "loss.{}".format(datetime.datetime.now())), 'w')
    best_accuracy_at_top_5 = 0
    epoch_duration = 0.0
    global_step = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:

            buf = '{} ********** Running {} on epoch({}/{}) ***********'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                                   phase, epoch+1, int(args.num_train_epochs))
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
            precision_ls = []
            start_time = time.time()
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(data_iter, desc="Iteration")):
                batch = {k: t.to(device) for k, t in batch.items()}

                input_ids = batch['input']
                visit_mask = batch['visit_mask']
                code_mask = batch['code_mask']
                label_task = batch['label_dx']
                labels_visit = batch['labels_visit']

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = model(input_ids,
                                     visit_mask,
                                     code_mask,
                                     label_task,
                                     labels_visit)
                        loss.backward()
                        optimizer.step()

                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                        # optimizer.zero_grad()
                        global_step += 1

                        fout.write("{}\n".format(loss.item()))
                        tr_loss += loss.item()
                        nb_tr_examples += input_ids.size(0)
                        nb_tr_steps += 1
                    else:
                        outputs = model(input_ids,
                                        visit_mask,
                                        code_mask)
                        predicts = outputs.cpu().detach().numpy()
                        trues = label_task.cpu().numpy()

                        recalls = Evaluation.visit_level_precision_at_k(trues, predicts)
                        accuracy = Evaluation.code_level_accuracy_at_k(trues, predicts)
                        precision_ls.append(recalls)
                        accuracy_ls.append(accuracy)

            duration = time.time() - start_time
            if phase == 'train':
                fout.write("train loss {} on epoch {}\n".format(epoch, tr_loss/nb_tr_steps))
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(output_dir, "pytorch_model.bin_{}".format(epoch))
                torch.save(model_to_save, output_model_file)
                buf = '{} {} Loss: {:.4f}, Duration: {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), phase, tr_loss/nb_tr_steps, duration)
                print2file(buf, log_file)
                epoch_duration += duration
            else:
                epoch_precision = (np.array(precision_ls)).mean(axis=0)
                buf = '{} {} Precision: {}, Duration: {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), phase, epoch_precision, duration)
                print2file(buf, log_file)
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
    torch.save(model_to_save, output_model_file)
    # torch.save(model_to_save.state_dict(), output_model_file)

    # Save the optimizer
    output_optimizer_file = os.path.join(output_dir, "pytorch_op.bin")
    torch.save(optimizer, output_optimizer_file)


if __name__ == "__main__":
    main()


