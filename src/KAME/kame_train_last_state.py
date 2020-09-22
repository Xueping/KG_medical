from __future__ import print_function
from __future__ import division
import torch.optim as optim
import pickle
from GRAM.gram_helpers import calculate_dimSize, get_rootCode, build_tree, print2file, build_tree_with_padding
from KAME.kame_helpers import leaf2ancestors
from KAME.dataset import FTDataset, load_data, collate
from KAME.kame_module import KAME
import torch
from torch.utils.data import DataLoader
import time
import copy
from tqdm import tqdm, trange
import random
import numpy as np
import argparse
import os
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
    parser.add_argument('--g_dim_size',
                        type=int,
                        default=200,
                        help="dimension size of graph layer")

    args = parser.parse_args()

    task = args.task + '_lr_' + str(args.learning_rate) + '_bs_' + str(args.train_batch_size) + \
           '_e_' + str(args.num_train_epochs) + '_ebd_' + str(args.embd_dim_size)
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
    # dir_path = '../../'
    if args.data_source == 'mimic':
        tree_file = dir_path + 'outputs/gram/data/mimic/mimic'
        seq_file = dir_path + 'outputs/gram/data/mimic/mimic.seqs'
        label_file = dir_path + 'outputs/gram/data/mimic/mimic.ccsSingleLevel.seqs'
    else:
        tree_file = dir_path + 'outputs/gram/data/eicu/eicu'
        seq_file = dir_path + 'outputs/gram/data/eicu/eicu_new.seqs'
        label_file = dir_path + 'outputs/gram/data/eicu/eicu.ccsSingleLevel.seqs'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    num_leaves = calculate_dimSize(seq_file)
    num_classes = calculate_dimSize(label_file)
    num_ancestors = get_rootCode(tree_file+'.level2.pk') - num_leaves + 1

    leaves_list = []
    ancestors_list = []
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

    internal_list = []
    internal_ancestors_list = []
    # for i in range(4, 0, -1):
    #     leaves, ancestors = build_tree(tree_file + '.a_level' + str(i) + '.pk')
    #     internal_list.extend(leaves)
    #     internal_ancestors_list.extend(ancestors)
    masks_ancestors_list = []
    for i in range(4, 0, -1):
        leaves, ancestors, masks = build_tree_with_padding(tree_file + '.a_level' + str(i) + '.pk', 5)
        internal_list.extend(leaves)
        internal_ancestors_list.extend(ancestors)
        masks_ancestors_list.extend(masks)
    internal_list = torch.tensor(internal_list).long().to(device)
    internal_ancestors_list = torch.tensor(internal_ancestors_list).long().to(device)
    masks_ancestors_list = torch.tensor(masks_ancestors_list).float().to(device)

    seqs = pickle.load(open(seq_file, 'rb'))
    labels = pickle.load(open(label_file, 'rb'))

    model = KAME(leaves_list, ancestors_list, masks_list,
                 internal_list, internal_ancestors_list, masks_ancestors_list,
                 num_leaves, num_ancestors, args.embd_dim_size, args.attn_dim_size,
                 args.rnn_dim_size, args.g_dim_size, num_classes, device)
    # Send the model to GPU
    model = model.to(device)
    # print(model)

    # Gather the parameters to be optimized/updated in this run. we will be updating all parameters.
    params_to_update = model.parameters()

    # Observe that all parameters are being optimized
    optimizer = optim.Adadelta(params_to_update)

    leaf2ans = leaf2ancestors(tree_file)
    print2file('Loading data ... ', log_file)
    train_set, valid_set, test_set = load_data(seqs, labels, leaf2ans, num_leaves)

    train_dataset = FTDataset(train_set[0], train_set[1], train_set[2])
    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                   collate_fn=lambda batch: collate(batch, num_classes, num_leaves),
                                   num_workers=0, shuffle=True)
    size_train_data = len(train_set[0])
    num_train_steps = int(size_train_data / args.train_batch_size * args.num_train_epochs)

    val_dataset = FTDataset(valid_set[0], valid_set[1], valid_set[2])
    val_data_loader = DataLoader(val_dataset, batch_size=args.train_batch_size,
                                 collate_fn=lambda batch: collate(batch, num_classes, num_leaves),
                                 num_workers=0, shuffle=True)
    size_val_data = len(valid_set[0])
    num_val_steps = int(size_val_data / args.train_batch_size * args.num_train_epochs)

    test_dataset = FTDataset(test_set[0], test_set[1], test_set[2])
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

    # Train and evaluate
    # train_model_last_state(model, data_dict, optimizer, device, args.train_batch_size, args.num_train_epochs,
    #                        num_leaves, num_classes, log_file, output_dir)
    # pickle.dump(hist, open(model_path+'/mimic.kame.val_loss_history', 'wb'), -1)

    print2file('\n', log_file)
    buf = '#####' * 10 + ' Training model ' + '#####' * 10
    print2file(buf, log_file)

    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy_at_top_5 = 0
    epoch_duration = 0.0

    for epoch in trange(args.num_train_epochs, desc="Epoch"):

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

            running_loss = 0.0
            precision_ls = []
            accuracy_ls = []

            data_iter = iter(data_dict[phase][0])

            start_time = time.time()
            # Iterate over data.
            for step, batch in enumerate(tqdm(data_iter, desc="Iteration")):
                batch = {k: t.to(device) for k, t in batch.items()}
                input = batch['input']
                input_ans = batch['input_ans']
                label_last = batch['label_last']
                mask = batch['mask']
                lengths = batch['lengths']

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in utils
                with torch.set_grad_enabled(phase == 'utils'):
                    # Get model outputs and calculate loss
                    _, outputs = model(input, input_ans, mask, lengths)

                    # Customise Loss function
                    logEps = 1e-8
                    cross_entropy = -(label_last * torch.log(outputs + logEps) +
                                      (1. - label_last) * torch.log(1. - outputs + logEps))
                    loglikelihood = cross_entropy.sum(axis=1)
                    loss = torch.mean(loglikelihood)

                    # backward + optimize only if in training phase
                    if phase == 'utils':
                        # loss.backward(retain_graph=True)
                        loss.backward()
                        optimizer.step()

                # statistics
                predicts = outputs.cpu().detach().numpy()
                trues = label_last.cpu().numpy()
                predicts = predicts.reshape(-1, predicts.shape[-1])
                trues = trues.reshape(-1, trues.shape[-1])

                precision = eval.visit_level_precision_at_k(trues, predicts)
                accuracy = eval.code_level_accuracy_at_k(trues, predicts)

                precision_ls.append(precision)
                accuracy_ls.append(accuracy)
            duration = time.time() - start_time

            epoch_precision = (np.array(precision_ls)).mean(axis=0)
            epoch_accuracy = (np.array(accuracy_ls)).mean(axis=0)

            buf = '{} {} ,Accuracy: {}, Duration: {}'.format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                phase, epoch_accuracy, duration)
            print2file(buf, log_file)

            buf = '{} {},Precision: {}, Duration: {}'.format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                phase, epoch_precision, duration)
            print2file(buf, log_file)

            # deep copy the model
            if phase == 'val' and epoch_accuracy[0] > best_accuracy_at_top_5:
                best_accuracy_at_top_5 = epoch_accuracy[0]
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'utils':
                epoch_duration += duration

        model_epoch_path = output_dir + '/kame_epoch(' + str(epoch) + ')_' + \
                           time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.model'
        buf = '{} Save the epoch({}) model to {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                         epoch, model_epoch_path)
        print2file(buf, log_file)
        torch.save(model, model_epoch_path)

    buf = '{} Training complete in {:.0f}m {:.0f}s'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                           epoch_duration // 60, epoch_duration % 60)
    print2file(buf, log_file)

    buf = '{} Best accuracy at top 5: {:4f}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                    best_accuracy_at_top_5)
    print2file(buf, log_file)

    # load best model weights
    model.load_state_dict(best_model_wts)

    model_file = output_dir + '/kame_' + str(args.num_train_epochs) + '_' + \
                 time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.model'
    buf = '{} Save the best model to {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), model_file)
    print2file(buf, log_file)
    torch.save(model, model_file)


if __name__ == "__main__":
    main()
