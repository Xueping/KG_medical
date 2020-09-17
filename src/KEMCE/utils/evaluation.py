from abc import ABCMeta
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, auc
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import random
from random import sample
from sklearn.manifold import TSNE
import heapq
import operator


class PredictionEvaluation(metaclass=ABCMeta):

    @staticmethod
    def metric_pred(y_true, probs, y_pred):
        [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
        # print(TN, FP, FN, TP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (FP + TN)
        precision = TP / (TP + FP)
        # recall = sensitivity
        sensitivity = recall = TP / (TP + FN)

        # calculate AUC
        roc_auc = roc_auc_score(y_true, probs)

        # calculate precision-recall curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, probs)
        # calculate F1 score
        f_score = f1_score(y_true, y_pred)
        # calculate precision-recall AUC
        pr_auc = auc(recall_curve, precision_curve)

        accuracy = round(accuracy, 4)
        precision = round(precision, 4)
        sensitivity = round(sensitivity, 4)
        specificity = round(specificity, 4)
        f_score = round(f_score, 4)
        pr_auc = round(pr_auc, 4)
        roc_auc = round(roc_auc, 4)

        return [accuracy, precision, sensitivity, specificity, f_score, pr_auc, roc_auc]

    @staticmethod
    def recall_top(y_true, y_pred, rank=[5, 10, 15, 20, 25, 30]):
        recall = list()
        for i in range(len(y_pred)):
            this_one = list()
            codes = y_true[i]
            tops = y_pred[i]
            length = len(set(codes))
            if length == 0:
                continue
            for rk in rank:
                if length > rk:
                    length = rk
                this_one.append(len(set(codes).intersection(set(tops[:rk])))*1.0/length)
            recall.append(this_one)
        return np.round((np.array(recall)).mean(axis=0), decimals=4), \
               np.round((np.array(recall)).std(axis=0), decimals=4)

    @staticmethod
    def visit_level_precision_at_k(y_true, y_pred, rank=[5, 10, 15, 20, 25, 30]):
        precision = list()
        for i in range(y_pred.shape[0]):
            this_one = list()
            # store indexes of the all ONE valves, e.g trues[i] = [1,0,1,1,0], then trueVec = [0, 2, 3]
            true_vec = [rk[0] for rk in
                       heapq.nlargest(np.count_nonzero(y_true[i]), enumerate(y_true[i]), key=operator.itemgetter(1))]
            length = len(true_vec)
            if length == 0:
                continue

            #  store indexes of the top 30 largest values, e.g. predicts[i] = [4,5,7,8,3],
            #  then preVec=[3, 2, 1, 0, 4]
            pre_vec = [rk[0] for rk in heapq.nlargest(30, enumerate(y_pred[i]), key=operator.itemgetter(1))]
            for rk in rank:
                tmp_length = min(rk, length)
                num_corrects = len(set(true_vec).intersection(set(pre_vec[:rk])))
                # print('K: hits -> {}:{}'.format(tmp_length, num_corrects))
                this_one.append(num_corrects*1.0/tmp_length)
            precision.append(this_one)

        return np.round((np.array(precision)).mean(axis=0), decimals=4)

    @staticmethod
    def code_level_accuracy_at_k(y_true, y_pred, rank=[5, 10, 15, 20, 25, 30]):
        precision = list()
        for i in range(y_pred.shape[0]):
            this_one = list()
            # store indexes of the all ONE valves, e.g trues[i] = [1,0,1,1,0], then trueVec = [0, 2, 3]
            true_vec = [rk[0] for rk in
                        heapq.nlargest(np.count_nonzero(y_true[i]), enumerate(y_true[i]), key=operator.itemgetter(1))]
            length = len(true_vec)
            if length == 0:
                continue

            #  store indexes of the top 30 largest values, e.g. predicts[i] = [4,5,7,8,3],
            #  then preVec=[3, 2, 1, 0, 4]
            pre_vec = [rk[0] for rk in heapq.nlargest(30, enumerate(y_pred[i]), key=operator.itemgetter(1))]
            for rk in rank:
                num_corrects = len(set(true_vec).intersection(set(pre_vec[:rk])))
                this_one.append(num_corrects * 1.0 / length)
            precision.append(this_one)

        return np.round((np.array(precision)).mean(axis=0), decimals=4)


    @staticmethod
    def code_level_top(y_true, y_pred, rank=[5, 10, 15, 20, 25, 30]):
        recall = list()
        for i in range(len(y_pred)):
            this_one = list()
            codes = y_true[i]
            tops = y_pred[i]
            for rk in rank:
                this_one.append(len(set(codes).intersection(set(tops[:rk]))) * 1.0 / rk)
            recall.append(this_one)
        return np.round((np.array(recall)).mean(axis=0), decimals=4).tolist()


class ConceptEvaluation(object):
    def __init__(self, reverse_dict, label4data):
        self.reverse_dict = reverse_dict
        self.label4data = label4data

    def get_clustering_nmi(self, embeddings):
        # embeddings = sess.run(self.model.final_weights)
        # get codes and index for diagnosis codes
        dx_codes = []
        index_codes = []  # index of valid diagnosis codes
        for k, v in self.reverse_dict.items():
            if v.startswith('D_'):
                dx_codes.append(v[2:])
                index_codes.append(int(k))

        # Label high level category (0:18) for each code
        dx_labels = np.empty(len(dx_codes), int)
        dx_index = 0
        for code in dx_codes:
            dx_labels[dx_index] = int(self.label4data.code2first_level_dx[code.replace('.', '')])
            dx_index += 1

        dx_weights = embeddings[index_codes]

        dx_uni_labels = np.unique(dx_labels).shape[0]
        k_means = KMeans(n_clusters=dx_uni_labels, random_state=42).fit(dx_weights)

        nmi = metrics.normalized_mutual_info_score(dx_labels, k_means.labels_)
        nmi_round = round(nmi, 4)

        return nmi_round

    def get_sample_tsne(self, embeddings, size_samples=2000):
        # embeddings = sess.run(self.model.final_weights)
        # get codes and index for diagnosis codes
        dx_codes = []
        index_codes = []  # index of valid diagnosis codes
        for k, v in self.reverse_dict.items():
            if v.startswith('D_'):
                dx_codes.append(v[2:])
                index_codes.append(int(k))

        # Label high level category (0:17) for each code
        dx_labels = np.empty(len(dx_codes), int)
        for dx_index, code in enumerate(dx_codes):
            dx_labels[dx_index] = int(self.label4data.code2first_level_dx[code.replace('.', '')])-1

        random.seed(42)
        samples = sample(list(zip(index_codes, dx_labels)), size_samples)
        indexes_samples = []
        labels_samples = []
        for index, label in samples:
            indexes_samples.append(index)
            labels_samples.append(label)

        dx_weights = embeddings[indexes_samples]
        tsne = TSNE(n_components=2)
        X_2d = tsne.fit_transform(dx_weights)

        return X_2d, labels_samples


    def get_nns_p_at_top_k(self, embeddings, top_k, valid_size=500):
        # similarity = sess.run(self.model.final_wgt_sim)
        # get codes and index for diagnosis codes
        dx_codes = []
        index_codes = []  # index of valid diagnosis codes
        for k, v in self.reverse_dict.items():
            if v.startswith('D_'):
                dx_codes.append(v[2:])
                index_codes.append(int(k))

        dx_weights = embeddings[index_codes]

        valid_index = list(range(valid_size))
        valid_weights = dx_weights[valid_index]
        similarity = np.matmul(valid_weights, np.transpose(dx_weights))

        total_precision = 0.
        for i in range(valid_size):
            valid_code = dx_codes[i]
            valid_label = self.label4data.code2first_level_dx[valid_code.replace('.', '')]
            nearest = (-similarity[i, :]).argsort()[1:top_k+1]
            no_same_cat = 0.
            actual_k = 0
            for k in range(top_k):
                close_word = dx_codes[nearest[k]]
                actual_k += 1
                close_label = self.label4data.code2first_level_dx[close_word.replace('.', '')]
                if valid_label == close_label:
                    no_same_cat += 1
            if actual_k > 0:
                total_precision += no_same_cat/actual_k

        evg_precision = round(total_precision/valid_size, 4)

        return evg_precision

    def get_nns_pairs_count(self):

        code_list = list(self.reverse_dict.values())[1:]
        dx_codes = list()
        tx_codes = list()
        for code in code_list:
            if code.startswith('D_'):
                dx_codes.append(code)
            else:
                tx_codes.append(code)

        label_cnt_dict = {}
        for dx in dx_codes:
            label = self.label4data.code2first_level_dx[dx[2:].replace('.', '')]
            if label in label_cnt_dict:
                label_cnt_dict[label] += 1
            else:
                label_cnt_dict[label] = 1

        no_cat = len(label_cnt_dict)
        total_pairs = 0
        for k, v in label_cnt_dict.items():
            total_pairs += v * (v-1)/2

        return no_cat, total_pairs
