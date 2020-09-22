from __future__ import print_function
from __future__ import division
import argparse
import os
import pickle
import torch
import time
import datetime
import copy
from tqdm import tqdm, trange
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from GRAM.gram_helpers import calculate_dimSize, get_rootCode, print2file, build_tree_with_padding
from GRAM.dataset import load_data, FTDataset, collate
from GRAM.gram_module import GRAM as gram_model, CrossEntropyLoss
from KEMCE.utils.evaluation import PredictionEvaluation as eval


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
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=1.0,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=100,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gpu",
                        default=0,
                        type=int,
                        help="CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--embd_dim_size',
                        type=int,
                        default=200,
                        help="dimension size of code embedding")
    parser.add_argument('--attn_dim_size',
                        type=int,
                        default=200,
                        help="dimension size of attention")
    parser.add_argument('--rnn_dim_size',
                        type=int,
                        default=200,
                        help="dimension size of rnn layer")

    args = parser.parse_args()

    task = args.task + '_lr_' + str(args.learning_rate) + '_bs_' + str(args.train_batch_size) + \
           '_epoch_' + str(args.num_train_epochs) + '_ebd_' + str(args.embd_dim_size)
    output_dir = os.path.join(args.output_dir, args.data_source, task)
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, args.data_source+'.log')
    buf = '{} seed:{}, gpu:{}, num_train_epochs:{}, learning_rate:{}, train_batch_size:{}, output_dir:{}'.format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        args.seed, args.gpu, int(args.num_train_epochs), args.learning_rate, args.train_batch_size,
        output_dir)
    print2file(buf, log_file)

    dir_path = args.data_dir
    if args.data_source == 'mimic':
        tree_file = dir_path + 'outputs/gram/data/mimic/mimic'
        seq_file = dir_path + 'outputs/gram/data/mimic/mimic.seqs'
        label_ccs_file = dir_path + 'outputs/gram/data/mimic/mimic.ccsSingleLevel.seqs'
    else:
        tree_file = dir_path + 'outputs/gram/data/eicu/eicu'
        seq_file = dir_path + 'outputs/gram/data/eicu/eicu_new.seqs'
        label_ccs_file = dir_path + 'outputs/gram/data/eicu/eicu.ccsSingleLevel.seqs'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    num_leaves = calculate_dimSize(seq_file)
    num_classes = calculate_dimSize(label_ccs_file)
    num_ancestors = get_rootCode(tree_file+'.level2.pk') - num_leaves + 1

    leaves_list = []
    ancestors_list = []
    # for i in range(5, 1, -1):
    # for i in range(5, 0, -1):
    #     leaves, ancestors = build_tree(tree_file + '.level' + str(i) + '.pk')
    #     leaves_list.extend(leaves)
    #     ancestors_list.extend(ancestors)

    masks_list = []
    for i in range(5, 0, -1):
        leaves, ancestors, masks = build_tree_with_padding(tree_file + '.level' + str(i) + '.pk')
        leaves_list.extend(leaves)
        ancestors_list.extend(ancestors)
        masks_list.extend(masks)
    leaves_list = torch.tensor(leaves_list).long().to(device)
    ancestors_list = torch.tensor(ancestors_list).long().to(device)
    masks_list = torch.tensor(masks_list).float().to(device)

    seqs = pickle.load(open(seq_file, 'rb'))
    labels = pickle.load(open(label_ccs_file, 'rb'))

    print2file('Loading data ... ', log_file)
    train_set, valid_set, test_set = load_data(seqs, labels)

    train_dataset = FTDataset(train_set[0], train_set[1])
    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                   collate_fn=lambda batch: collate(batch, num_classes, num_leaves),
                                   num_workers=0, shuffle=True)
    size_train_data = len(train_set[0])
    num_train_steps = int(size_train_data / args.train_batch_size * args.num_train_epochs)

    val_dataset = FTDataset(valid_set[0], valid_set[1])
    val_data_loader = DataLoader(val_dataset, batch_size=args.train_batch_size,
                                 collate_fn=lambda batch: collate(batch, num_classes, num_leaves),
                                 num_workers=0, shuffle=True)
    size_val_data = len(valid_set[0])
    num_val_steps = int(size_val_data / args.train_batch_size * args.num_train_epochs)

    test_dataset = FTDataset(test_set[0], test_set[1])
    test_data_loader = DataLoader(test_dataset, batch_size=args.train_batch_size,
                                  collate_fn=lambda batch: collate(batch, num_classes, num_leaves),
                                  num_workers=0, shuffle=True)
    size_test_data = len(test_set[0])
    num_test_steps = int(size_test_data / args.train_batch_size * args.num_train_epochs)

    data_dict = dict()
    data_dict['utils'] = [train_data_loader, size_train_data, num_train_steps]
    data_dict['val'] = [val_data_loader, size_val_data, num_val_steps]
    data_dict['test'] = [test_data_loader, size_test_data, num_test_steps]
    print('done!!')

    model = gram_model(leaves_list, ancestors_list, masks_list,
                       num_leaves, num_ancestors, args.embd_dim_size,
                       args.attn_dim_size, args.rnn_dim_size, num_classes, device)
    # Send the model to GPU
    model = model.to(device)
    # print(model)
    print(model)

    # Gather the parameters to be optimized/updated in this run. we will be updating all parameters.
    params_to_update = model.parameters()
    # Observe that all parameters are being optimized
    optimizer = optim.Adadelta(params_to_update)
    loss_fct = CrossEntropyLoss()

    print2file('\n', log_file)
    buf = '#####' * 10 + ' Training model ' + '#####' * 10
    print2file(buf, log_file)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy_at_top_5 = 0
    epoch_duration = 0.0
    global_step = 0
    # loss_fct = CrossEntropyLoss()

    fout = open(os.path.join(output_dir, "loss.{}".format(datetime.datetime.now())), 'w')

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

        buf = '{} Epoch {}/{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                      epoch, args.num_train_epochs - 1)
        print2file(buf, log_file)
        buf = '-' * 10
        print2file(buf, log_file)

        # Each epoch has a training and validation phase
        for phase in ['utils', 'val', 'test']:
            if phase == 'utils':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            buf = '{} ********** Running {} on epoch({}/{}) ***********'.format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), phase, epoch + 1, int(args.num_train_epochs))
            print2file(buf, log_file)
            buf = '{} Num examples = {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                data_dict[phase][1])
            print2file(buf, log_file)
            buf = '{}  Num steps = {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                              data_dict[phase][2])
            print2file(buf, log_file)

            data_iter = iter(data_dict[phase][0])
            tr_loss = 0
            accuracy_ls = []
            precision_ls = []
            start_time = time.time()
            nb_tr_examples, nb_tr_steps = 0, 0
            # Iterate over data.
            for step, batch in enumerate(tqdm(data_iter, desc="Iteration")):
                batch = {k: t.to(device) for k, t in batch.items()}
                input = batch['input']
                label_last = batch['label_last']
                mask = batch['mask']
                lengths = batch['lengths']

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward and track history if only in utils
                with torch.set_grad_enabled(phase == 'utils'):
                    # backward + optimize only if in training phase
                    _, last_valid_output = model(input, mask, lengths)
                    if phase == 'utils':
                        loss = loss_fct(last_valid_output, label_last)
                        # print(loss)
                        # loss.backward(retain_graph=True)
                        loss.backward()
                        optimizer.step()

                        tr_loss += loss.item()
                        nb_tr_examples += data_dict[phase][1]
                        nb_tr_steps += 1
                    else:
                        # statistics
                        predicts = last_valid_output.cpu().detach().numpy()
                        trues = label_last.cpu().numpy()
                        predicts = predicts.reshape(-1, predicts.shape[-1])
                        trues = trues.reshape(-1, trues.shape[-1])

                        precision = eval.visit_level_precision_at_k(trues, predicts)
                        accuracy = eval.code_level_accuracy_at_k(trues, predicts)

                        precision_ls.append(precision)
                        accuracy_ls.append(accuracy)

            duration = time.time() - start_time
            if phase == 'utils':
                fout.write("train loss {} on epoch {}\n".format(epoch, tr_loss / nb_tr_steps))
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(output_dir, "pytorch_model.bin_{}".format(epoch))
                torch.save(model_to_save, output_model_file)
                buf = '{} {} Loss: {:.4f}, Duration: {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), phase, tr_loss / nb_tr_steps, duration)
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

    # buf = '{} Best val Loss: {:4f}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), best_loss)
    # print2file(buf, log_file)
    buf = '{} Best accuracy at top 5: {:4f}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                    best_accuracy_at_top_5)
    print2file(buf, log_file)

    # load best model weights
    model.load_state_dict(best_model_wts)

    model_file = output_dir + '/gram_' + str(args.num_train_epochs) + '_' + \
                 time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.model'
    buf = '{} Save the best model to {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), model_file)
    print2file(buf, log_file)
    torch.save(model, model_file)


if __name__ == "__main__":
    main()

