#################################################################
# Code written by Xueping Peng according to GRAM paper and original codes
#################################################################
from __future__ import print_function
from __future__ import division
import random
import torch
import pickle
import numpy as np
import time
import copy
from GRAM.gram_helpers import print2file
from KEMCE.utils.evaluation import PredictionEvaluation as eval

_TEST_RATIO = 0.15
_VALIDATION_RATIO = 0.1


def pad_matrix(seqs, ans_seqs, labels, num_leaf_code, num_class):
    lengths = np.array([len(seq) for seq in seqs]) - 1
    lengths_ans = []
    for seq in ans_seqs:
        for visit in seq:
            lengths_ans.append(len(visit))
    maxlen_ans = np.max(np.array(lengths_ans))
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((n_samples, maxlen, num_leaf_code)).astype(np.float32)
    f = np.zeros((n_samples, maxlen, maxlen_ans)).astype(np.long)
    y = np.zeros((n_samples, maxlen, num_class)).astype(np.float32)
    mask = np.zeros((n_samples, maxlen)).astype(np.float32)

    for idx, (seq, ans_seq, lseq) in enumerate(zip(seqs,ans_seqs,labels)):
        for xvec, subseq in zip(x[idx,:,:], seq[:-1]):
            xvec[subseq] = 1.
        for fvec, subseq in zip(f[idx,:,:], ans_seq[:-1]):
            fvec[:len(subseq)] = subseq
        for yvec, subseq in zip(y[idx, :, :], lseq[1:]):
            yvec[subseq] = 1.
        mask[idx, :lengths[idx]] = 1.

    lengths = np.array(lengths, dtype=np.float32)
    return x, f, y, mask, lengths


def leaf2ancestors(tree_path):
    ancestors = {}
    for i in range(5, 0, -1):
        tree_file = tree_path + '.level' + str(i) + '.pk'
        tree_map = pickle.load(open(tree_file, 'rb'))
        ancestors.update(tree_map)
    return ancestors


def codes2ancestors(seq, leaf2ans, num_leaves):
    new_seq = []
    for visit in seq:
        ans_ls = []
        for code in visit:
            ans = leaf2ans[code][1:]
            ans_ls.extend(ans)
        ans_ls = list(set(ans_ls))
        #  the first element for padding
        ans_ls = [ans-num_leaves+1 for ans in ans_ls]
        new_seq.append(ans_ls)
    return new_seq


def load_data(sequences, labels, leaf2ans, num_leaves):
    np.random.seed(0)
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
    test_set =  (test_set_x, test_set_f, test_set_y)

    return train_set, valid_set, test_set


def train_model(model, data_dict, optimizer, device, batch_size, num_epochs,
                num_leaves, num_classes, log_file, model_path):

    print2file('\n', log_file)
    buf = '#####' * 10 + ' Training model '+ '#####' * 10
    print2file(buf, log_file)

    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e+10
    best_accuracy_at_top_5 = 0
    epoch_duration = 0.0

    for epoch in range(num_epochs):

        buf = '{} Epoch {}/{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), epoch, num_epochs - 1)
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
            precision_lst = []
            accuracy_ls = []

            data_set = data_dict[phase]
            n_batches = int(np.ceil(float(len(data_set[0])) / float(batch_size)))

            start_time = time.time()
            # Iterate over data.
            for index in random.sample(range(n_batches), n_batches):
                batchX = data_set[0][index * batch_size:(index + 1) * batch_size]
                batchF = data_set[1][index * batch_size:(index + 1) * batch_size]
                batchY = data_set[2][index * batch_size:(index + 1) * batch_size]
                x, f, y, mask, lengths = pad_matrix(batchX, batchF, batchY, num_leaves, num_classes)

                batch_x = torch.from_numpy(x).to(device)
                batch_f = torch.from_numpy(f).to(device)
                batch_y = torch.from_numpy(y).to(device)
                lengths = torch.from_numpy(lengths).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in utils
                with torch.set_grad_enabled(phase == 'utils'):
                    # Get model outputs and calculate loss
                    outputs = model(batch_x, batch_f, mask)

                    # Customise Loss function
                    logEps = 1e-8
                    cross_entropy = -(batch_y * torch.log(outputs + logEps) +
                                      (1. - batch_y) * torch.log(1. - outputs + logEps))
                    loglikelihood = cross_entropy.sum(axis=2).sum(axis=1) / lengths
                    loss = torch.mean(loglikelihood)

                    # backward + optimize only if in training phase
                    if phase == 'utils':
                        loss.backward(retain_graph=True)
                        # loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                predicts = outputs.cpu().detach().numpy()
                trues = batch_y.cpu().numpy()
                predicts = predicts.reshape(-1, predicts.shape[-1])
                trues = trues.reshape(-1, trues.shape[-1])

                # recalls = eval.visit_level_precision_at_k(trues, predicts)
                accuracy = eval.code_level_accuracy_at_k(trues, predicts)

                # precision_lst.append(recalls)
                accuracy_ls.append(accuracy)
            duration = time.time() - start_time
            epoch_loss = running_loss / n_batches

            # epoch_precision = (np.array(precision_lst)).mean(axis=0)
            epoch_accuracy = (np.array(accuracy_ls)).mean(axis=0)

            # buf = '{} {} Loss: {:.4f}, Duration: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            #                                                 phase, epoch_loss, duration)
            # print2file(buf, log_file)
            buf = '{} {} Loss: {:.4f},Accuracy: {}, Duration: {}'.format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                phase, epoch_loss, epoch_accuracy, duration)
            print2file(buf, log_file)

            # deep copy the model
            if phase == 'val' and epoch_accuracy[0] > best_accuracy_at_top_5:
            # if phase == 'val' and epoch_loss < best_loss:
            #     best_loss = epoch_loss
                best_accuracy_at_top_5 = epoch_accuracy[0]
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            if phase == 'utils':
                epoch_duration += duration

        model_epoch_path = model_path + '/mimic.kame_epoch(' + str(epoch) + ')_' + \
                     time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.model'
        buf = '{} Save the epoch({}) model to {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                         epoch, model_epoch_path)
        print2file(buf, log_file)
        torch.save(model, model_epoch_path)

    buf = '{} Training complete in {:.0f}m {:.0f}s'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), epoch_duration // 60, epoch_duration % 60)
    print2file(buf, log_file)

    # buf = '{} Best val Loss: {:4f}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), best_loss)
    # print2file(buf, log_file)

    buf = '{} Best accuracy at top 5: {:4f}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), best_accuracy_at_top_5)
    print2file(buf, log_file)

    # load best model weights
    model.load_state_dict(best_model_wts)

    model_file = model_path + '/mimic.kame_' + str(num_epochs) + '_' + \
                 time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.model'
    buf = '{} Save the best model to {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), model_file)
    print2file(buf, log_file)
    torch.save(model, model_file)

    return model, val_loss_history


def eval_model(model, data_dict, device, batch_size, num_leaves, num_classes, log_file):

    print2file('\n', log_file)
    buf = '#####' * 10 + ' Testing model ' + '#####' * 10
    print2file(buf, log_file)

    test_duration = 0.0

    model.eval()  # Set model to evaluate mode

    running_loss = 0.0

    data_set = data_dict['test']
    n_batches = int(np.ceil(float(len(data_set[0])) / float(batch_size)))

    start_time = time.time()
    precision_lst = []
    accuracy_ls = []
    # Iterate over data.
    for index in random.sample(range(n_batches), n_batches):
        batchX = data_set[0][index * batch_size:(index + 1) * batch_size]
        batchF = data_set[1][index * batch_size:(index + 1) * batch_size]
        batchY = data_set[2][index * batch_size:(index + 1) * batch_size]
        x, f, y, mask, lengths = pad_matrix(batchX, batchF, batchY, num_leaves, num_classes)

        batch_x = torch.from_numpy(x).to(device)
        batch_f = torch.from_numpy(f).to(device)
        batch_y = torch.from_numpy(y).to(device)
        lengths = torch.from_numpy(lengths).to(device)

        # forward
        # track history if only in utils
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(batch_x, batch_f, mask)

            # Customise Loss function
            logEps = 1e-8
            cross_entropy = -(batch_y * torch.log(outputs + logEps) +
                              (1. - batch_y) * torch.log(1. - outputs + logEps))
            loglikelihood = cross_entropy.sum(axis=2).sum(axis=1) / lengths
            loss = torch.mean(loglikelihood)

            predicts = outputs.cpu().numpy()
            trues = batch_y.cpu().numpy()
            predicts = predicts.reshape(-1, predicts.shape[-1])
            trues = trues.reshape(-1, trues.shape[-1])

            recalls = eval.visit_level_precision_at_k(trues, predicts)
            accuracys = eval.code_level_accuracy_at_k(trues, predicts)

            precision_lst.append(recalls)
            accuracy_ls.append(accuracys)
        # statistics
        running_loss += loss.item()

    duration = time.time() - start_time
    test_loss = running_loss / n_batches
    final_recall = (np.array(precision_lst)).mean(axis=0)
    final_accuracy = (np.array(accuracy_ls)).mean(axis=0)

    buf = '{} Testing complete in {:.0f}m {:.0f}s'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                          duration // 60, duration % 60)
    print2file(buf, log_file)

    buf = '{} test Loss: {:4f}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), test_loss)
    print2file(buf, log_file)
    buf = '{} Precision: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), final_recall)
    print2file(buf, log_file)
    buf = '{} Accuracy: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), final_accuracy)
    print2file(buf, log_file)


