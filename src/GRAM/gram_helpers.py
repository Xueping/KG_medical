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
from KEMCE.utils.evaluation import PredictionEvaluation as eval

_TEST_RATIO = 0.15
_VALIDATION_RATIO = 0.1


def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4] + '.' + dxStr[4:]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3] + '.' + dxStr[3:]
        else:
            return dxStr


def convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3]
        else:
            return dxStr


def print2file(buf, outFile):
    print(buf)
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()


def pad_matrix(seqs, labels, num_leaf_code, num_class):
    lengths = np.array([len(seq) for seq in seqs]) - 1
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((n_samples, maxlen, num_leaf_code)).astype(np.float32)
    y = np.zeros((n_samples, maxlen, num_class)).astype(np.float32)
    last_y = np.zeros((n_samples, num_class)).astype(np.float32)
    mask = np.zeros((n_samples, maxlen)).astype(np.float32)

    for idx, (seq, lseq) in enumerate(zip(seqs,labels)):
        for xvec, subseq in zip(x[idx,:,:], seq[:-1]):
            xvec[subseq] = 1.
        for yvec, subseq in zip(y[idx,:,:], lseq[1:]):
            yvec[subseq] = 1.
        mask[idx, :lengths[idx]] = 1.
        last_y[idx] = y[idx, lengths[idx]-1, :]

    lengths = np.array(lengths, dtype=np.float32)
    return x, y, mask, lengths, last_y


def calculate_dimSize(seqFile):
    seqs = pickle.load(open(seqFile, 'rb'))
    codeSet = set()
    for patient in seqs:
        for visit in patient:
            for code in visit:
                codeSet.add(code)
    return max(codeSet) + 1


def get_rootCode(treeFile):
    tree = pickle.load(open(treeFile, 'rb'))
    rootCode = list(tree.values())[0][1]
    return rootCode


def build_tree(treeFile):
    treeMap = pickle.load(open(treeFile, 'rb'))
    if len(treeMap) == 0:
        return [], []
    ancestors = np.array(list(treeMap.values())).astype('int32')
    ancSize = ancestors.shape[1]
    leaves = []
    for k in treeMap.keys():
        leaves.append([k] * ancSize)
    leaves = np.array(leaves).astype('int32')
    return leaves, ancestors


def build_tree_with_padding(treeFile):
    max_len_ancestors = 6 # the max length of code's ancestors including itself
    treeMap = pickle.load(open(treeFile, 'rb'))
    if len(treeMap) == 0:
        return [], [], []
    ancestors = np.array(list(treeMap.values())).astype('int32')
    ancSize = ancestors.shape[1]
    leaves = []
    for k in treeMap.keys():
        leaves.append([k] * ancSize)
    leaves = np.array(leaves).astype('int32')
    # add padding and mask
    if ancSize < 6:
        ones = np.ones((ancestors.shape[0], ancSize)).astype('int32')
        zeros = np.zeros((ancestors.shape[0], max_len_ancestors-ancSize)).astype('int32')
        leaves = np.concatenate([leaves, zeros], axis=1)
        ancestors = np.concatenate([ancestors, zeros], axis=1)
        mask = np.concatenate([ones, zeros], axis=1)
    else:
        mask = np.ones((ancestors.shape[0], 6))
    return leaves, ancestors, mask


def load_data(sequences, labels):

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
            accuracy_ls = []

            data_set = data_dict[phase]
            n_batches = int(np.ceil(float(len(data_set[0])) / float(batch_size)))

            start_time = time.time()
            # Iterate over data.
            for index in random.sample(range(n_batches), n_batches):
                batchX = data_set[0][index * batch_size:(index + 1) * batch_size]
                batchY = data_set[1][index * batch_size:(index + 1) * batch_size]
                x, y, mask, lengths, _ = pad_matrix(batchX, batchY, num_leaves, num_classes)

                batchX = torch.from_numpy(x).to(device)
                batchY = torch.from_numpy(y).to(device)
                lengths = torch.from_numpy(lengths).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in utils
                with torch.set_grad_enabled(phase == 'utils'):
                    # Get model outputs and calculate loss
                    outputs, last_valid_output = model(batchX, mask, lengths)

                    # Customise Loss function
                    logEps = 1e-8
                    cross_entropy = -(batchY * torch.log(outputs + logEps) +
                                      (1. - batchY) * torch.log(1. - outputs + logEps))
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
                trues = batchY.cpu().numpy()
                predicts = predicts.reshape(-1, predicts.shape[-1])
                trues = trues.reshape(-1, trues.shape[-1])

                # recalls = eval.visit_level_precision_at_k(trues, predicts)
                accuracy = eval.code_level_accuracy_at_k(trues, predicts)

                # precision_lst.append(recalls)
                accuracy_ls.append(accuracy)

            # epoch_precision = (np.array(precision_lst)).mean(axis=0)
            epoch_accuracy = (np.array(accuracy_ls)).mean(axis=0)
            duration = time.time() - start_time
            epoch_loss = running_loss / n_batches

            buf = '{} {} Loss: {:.4f},Accuracy: {}, Duration: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                            phase, epoch_loss,epoch_accuracy, duration)
            print2file(buf, log_file)

            # deep copy the model
            # if phase == 'val' and epoch_loss < best_loss:
            #     best_loss = epoch_loss
            if phase == 'val' and epoch_accuracy[0] > best_accuracy_at_top_5:
                best_accuracy_at_top_5 = epoch_accuracy[0]
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            if phase == 'utils':
                epoch_duration += duration

        model_epoch_file = model_path + '/mimic.gram_epoch(' + str(epoch) + ')_' + \
                           time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.model'
        buf = '{} Save the epock({}) model to {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                         epoch, model_epoch_file)
        print2file(buf, log_file)
        torch.save(model, model_epoch_file)

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

    model_file = model_path + '/mimic.gram_' + str(num_epochs) + '_' + \
                 time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.model'
    buf = '{} Save the best model to {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), model_file)
    print2file(buf, log_file)
    torch.save(model, model_file)

    return model, val_loss_history


def train_model_last_state(model, data_dict, optimizer, device, batch_size, num_epochs,
                           num_leaves, num_classes, log_file, model_path):
    print2file('\n', log_file)
    buf = '#####' * 10 + ' Training model ' + '#####' * 10
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
            accuracy_ls = []

            data_set = data_dict[phase]
            n_batches = int(np.ceil(float(len(data_set[0])) / float(batch_size)))

            start_time = time.time()
            # Iterate over data.
            for index in random.sample(range(n_batches), n_batches):
                batchX = data_set[0][index * batch_size:(index + 1) * batch_size]
                batchY = data_set[1][index * batch_size:(index + 1) * batch_size]
                x, y, mask, lengths, last_y = pad_matrix(batchX, batchY, num_leaves, num_classes)

                batchX = torch.from_numpy(x).to(device)
                batchY = torch.from_numpy(last_y).to(device)
                lengths = torch.from_numpy(lengths).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in utils
                with torch.set_grad_enabled(phase == 'utils'):
                    # Get model outputs and calculate loss
                    _, last_valid_output = model(batchX, mask, lengths)

                    # Customise Loss function
                    logEps = 1e-8
                    cross_entropy = -(batchY * torch.log(last_valid_output + logEps) +
                                      (1. - batchY) * torch.log(1. - last_valid_output + logEps))
                    loglikelihood = cross_entropy.sum(axis=1)
                    loss = torch.mean(loglikelihood)

                    # backward + optimize only if in training phase
                    if phase == 'utils':
                        loss.backward(retain_graph=True)
                        # loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                predicts = last_valid_output.cpu().detach().numpy()
                trues = batchY.cpu().numpy()
                predicts = predicts.reshape(-1, predicts.shape[-1])
                trues = trues.reshape(-1, trues.shape[-1])

                # recalls = eval.visit_level_precision_at_k(trues, predicts)
                accuracy = eval.code_level_accuracy_at_k(trues, predicts)

                # precision_lst.append(recalls)
                accuracy_ls.append(accuracy)

            # epoch_precision = (np.array(precision_lst)).mean(axis=0)
            epoch_accuracy = (np.array(accuracy_ls)).mean(axis=0)
            duration = time.time() - start_time
            epoch_loss = running_loss / n_batches

            buf = '{} {} Loss: {:.4f},Accuracy: {}, Duration: {}'.format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                phase, epoch_loss, epoch_accuracy, duration)
            print2file(buf, log_file)

            # deep copy the model
            # if phase == 'val' and epoch_loss < best_loss:
            #     best_loss = epoch_loss
            if phase == 'val' and epoch_accuracy[0] > best_accuracy_at_top_5:
                best_accuracy_at_top_5 = epoch_accuracy[0]
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            if phase == 'utils':
                epoch_duration += duration

        model_epoch_file = model_path + '/mimic.gram_epoch(' + str(epoch) + ')_' + \
                           time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.model'
        buf = '{} Save the epock({}) model to {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                         epoch, model_epoch_file)
        print2file(buf, log_file)
        torch.save(model, model_epoch_file)

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

    model_file = model_path + '/mimic.gram_' + str(num_epochs) + '_' + \
                 time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.model'
    buf = '{} Save the best model to {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), model_file)
    print2file(buf, log_file)
    torch.save(model, model_file)

    return model, val_loss_history


def eval_model(model, data_dict, device, batch_size, num_leaves, num_classes, log_file):

    print2file('\n', log_file)
    buf = '#####' * 10 + ' Testing model ' + '#####' * 10
    print2file(buf, log_file)

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
        batchY = data_set[1][index * batch_size:(index + 1) * batch_size]
        x, y, mask, lengths = pad_matrix(batchX, batchY, num_leaves, num_classes)

        batchX = torch.from_numpy(x).to(device)
        batchY = torch.from_numpy(y).to(device)
        lengths = torch.from_numpy(lengths).to(device)

        # forward
        # track history if only in utils
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(batchX, mask)

            # Customise Loss function
            logEps = 1e-8
            cross_entropy = -(batchY * torch.log(outputs + logEps) +
                              (1. - batchY) * torch.log(1. - outputs + logEps))
            loglikelihood = cross_entropy.sum(axis=2).sum(axis=1) / lengths
            loss = torch.mean(loglikelihood)

            predicts = outputs.cpu().numpy()
            trues = batchY.cpu().numpy()

            predicts = predicts.reshape(-1, predicts.shape[-1])
            trues = trues.reshape(-1, trues.shape[-1])
            precision = eval.visit_level_precision_at_k(trues, predicts)
            accuracys = eval.code_level_accuracy_at_k(trues, predicts)
            precision_lst.append(precision)
            accuracy_ls.append(accuracys)
        # statistics
        running_loss += loss.item()

    duration = time.time() - start_time
    test_loss = running_loss / n_batches
    final_precision = (np.array(precision_lst)).mean(axis=0)
    final_accuracy = (np.array(accuracy_ls)).mean(axis=0)

    buf = '{} Testing complete in {:.0f}m {:.0f}s'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                          duration // 60, duration % 60)
    print2file(buf, log_file)

    buf = '{} test Loss: {:4f}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), test_loss)
    print2file(buf, log_file)
    buf = '{} precision@[5, 10, 15, 20, 25, 30]: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), final_precision)
    print2file(buf, log_file)
    buf = '{} Accuracy: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), final_accuracy)
    print2file(buf, log_file)

