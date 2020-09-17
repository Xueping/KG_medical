from datetime import datetime
import pickle
import pandas as pd
from GRAM.gram_helpers import convert_to_3digit_icd9
from KEMCE.dataset import LabelsForData


def time_spans(interval):
    sep = '[SPAN'
    if interval <= 30:
        sep += '1M]'
    elif 30 < interval <= 90:
        sep += '3M]'
    elif 90 < interval <= 180:
        sep += '6M]'
    elif 180 < interval <= 365:
        sep += '12M]'
    else:
        sep += '12M+]'
    return sep


def mimic_processing_for_mort_los(adm_file, dx_file, out_file):

    '''
    The processing for visit level task, such as mortality and length of stay
    :param adm_file: admission file
    :param dx_file: diagnosis file
    :param out_file:
    :return:
    '''

    print('Building admission-label mapping')
    # admLabelMap = {}  # admissionID: [mortality, los]
    #     # infd = open(adm_file, 'r')
    #     # infd.readline()
    #     # for line in infd:
    #     #     tokens = line.strip().split(',')
    #     #     admId = int(tokens[2])
    #     #     admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
    #     #     sepTime = datetime.strptime(tokens[4], '%Y-%m-%d %H:%M:%S')
    #     #     mortality = int(tokens[18])
    #     #     interval = (sepTime - admTime).days + 1
    #     #     los = 0 if interval <= 7 else 1
    #     #     admLabelMap[admId] = [mortality, los]
    #     # infd.close()

    adm = pd.read_csv(adm_file, header=0)
    admLabelMap = {}  # admissionID: [mortality, los]
    for index, row in adm.iterrows():
        #     print(row.HADM_ID, row.ADMITTIME, row.DISCHTIME,row.HOSPITAL_EXPIRE_FLAG )
        admId = int(row.HADM_ID)
        admTime = datetime.strptime(row.ADMITTIME, '%Y-%m-%d %H:%M:%S')
        sepTime = datetime.strptime(row.DISCHTIME, '%Y-%m-%d %H:%M:%S')
        mortality = int(row.HOSPITAL_EXPIRE_FLAG)
        interval = (sepTime - admTime).days + 1
        los = 0 if interval <= 7 else 1
        admLabelMap[admId] = [mortality, los]

    print('Building admission-dxList mapping')
    admDxMap = {}  # admissionID: [code1, code2, ....]
    infd = open(dx_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        dx = tokens[4][1:-1]
        if len(dx) == 0:
            continue

        dxStr = 'D_' + dx

        if admId in admDxMap:
            admDxMap[admId].append(dxStr)
        else:
            admDxMap[admId] = [dxStr]
    infd.close()

    print('Building dxList-label mapping')
    visits = []
    labels = []
    for admId, codes in admDxMap.items():
        code_str = '[CLS] ' + ' '.join(codes) + ' [SEP]'
        label = admLabelMap[admId]

        visits.append(code_str)
        labels.append(label)

    pickle.dump(visits, open(out_file+'.visits', 'wb'), -1)
    pickle.dump(labels, open(out_file+'.labels', 'wb'), -1)


def mimic_processing_for_readm_dx(adm_file, dx_file, single_dx_file, multi_dx_file, out_file):

    label4data = LabelsForData(multi_dx_file, single_dx_file)
    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = open(adm_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]
    infd.close()

    print('Building admission-dxList mapping')
    admDxMap = {}
    admDxMap_3digit = {}
    admDxMap_ccs = {}
    infd = open(dx_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        dx = tokens[4][1:-1]
        if len(dx) == 0:
            continue

        dxStr = 'D_' + dx
        dxStr_3digit = 'D_' + convert_to_3digit_icd9(dx)
        dxStr_ccs_single = 'D_' + label4data.code2single_dx[dx]

        if admId in admDxMap:
            admDxMap[admId].append(dxStr)
        else:
            admDxMap[admId] = [dxStr]

        if admId in admDxMap_3digit:
            admDxMap_3digit[admId].append(dxStr_3digit)
        else:
            admDxMap_3digit[admId] = [dxStr_3digit]

        if admId in admDxMap_ccs:
            admDxMap_ccs[admId].append(dxStr_ccs_single)
        else:
            admDxMap_ccs[admId] = [dxStr_ccs_single]
    infd.close()

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    pidSeqMap_3digit = {}
    pidSeqMap_ccs = {}
    for pid, admIdList in pidAdmMap.items():
        new_admIdList = []
        for admId in admIdList:
            if admId in admDxMap:
                new_admIdList.append(admId)
        if len(new_admIdList) < 2: continue
        # print(admIdList)
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in new_admIdList])
        pidSeqMap[pid] = sortedList

        sortedList_3digit = sorted([(admDateMap[admId], admDxMap_3digit[admId]) for admId in new_admIdList])
        pidSeqMap_3digit[pid] = sortedList_3digit

        sortedList_ccs = sorted([(admDateMap[admId], admDxMap_ccs[admId]) for admId in new_admIdList])
        pidSeqMap_ccs[pid] = sortedList_ccs

    print('Building strSeqs, span label')
    seqs = []
    seqs_span = []
    for pid, visits in pidSeqMap.items():
        sep = []
        spans = []
        first_time = visits[0][0]
        for i, visit in enumerate(visits):
            current_time = visit[0]
            interval = (current_time - first_time).days
            first_time = current_time
            sep.append(visit[1])
            span_flag = 0 if interval <= 30 else 1
            spans.append(span_flag)
        seqs.append(sep)
        seqs_span.append(spans)

    print('Building pids, dates, strSeqs for 3digit ICD9 code')
    seqs_3digit = []
    for pid, visits in pidSeqMap_3digit.items():
        seq = []
        for visit in visits:
            seq.append(visit[1])
        seqs_3digit.append(seq)

    print('Converting strSeqs to intSeqs, and making types for 3digit ICD9 code')
    dict_3digit = {}
    newSeqs_3digit = []
    for patient in seqs_3digit:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in set(visit):
                if code in dict_3digit:
                    newVisit.append(dict_3digit[code])
                else:
                    dict_3digit[code] = len(dict_3digit)
                    newVisit.append(dict_3digit[code])
            newPatient.append(newVisit)
        newSeqs_3digit.append(newPatient)

    print('Building strSeqs for CCS single-level code')
    seqs_ccs = []
    for pid, visits in pidSeqMap_ccs.items():
        seq = []
        for visit in visits:
            seq.append(visit[1])
        seqs_ccs.append(seq)

    print('Converting strSeqs to intSeqs, and making types for ccs single-level code')
    dict_ccs = {}
    newSeqs_ccs = []
    for patient in seqs_ccs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in set(visit):
                if code in dict_ccs:
                    newVisit.append(dict_ccs[code])
                else:
                    dict_ccs[code] = len(dict_ccs)
                    newVisit.append(dict_ccs[code])
            newPatient.append(newVisit)
        newSeqs_ccs.append(newPatient)

    print('Converting seqs to BERT inputs')

    inputs_all = []
    labels_all = []
    max_visit_len = 0
    max_seqs_len = 0
    for i, seq in enumerate(seqs):
        length = len(seq)
        if length >= 11:
            last_seqs = seq[length-11:]
            last_spans = seqs_span[i][length-11:]
            last_seq_3digit = newSeqs_3digit[i][length-11:]
            last_seq_ccs = newSeqs_ccs[i][length - 11:]
        else:
            last_seqs = seq
            last_spans = seqs_span[i]
            last_seq_3digit = newSeqs_3digit[i]
            last_seq_ccs = newSeqs_ccs[i]

        if len(last_seqs) > max_visit_len:
            max_visit_len = len(last_seqs)

        for index in range(len(last_seqs)-1):
            label_span = last_spans[index+1]
            label_3digit = last_seq_3digit[index + 1]
            label_ccs = last_seq_ccs[index + 1]
            input_str = '[CLS] '
            for sub_i in range(index+1):
                input_str += ' '.join(last_seqs[sub_i]) + ' [SEP] '
            input_str = input_str[:-1]
            str_length = len(input_str.split(' '))
            if str_length > max_seqs_len:
                max_seqs_len = str_length
            inputs_all.append(input_str)
            labels_all.append([label_span, label_3digit, label_ccs])
    max_visit_len -= 1
    pickle.dump(inputs_all, open(out_file + '.inputs_all.seqs', 'wb'), -1)
    pickle.dump(labels_all, open(out_file + '.labels_all.label', 'wb'), -1)
    pickle.dump(dict_3digit, open(out_file + '.3digitICD9.dict', 'wb'), -1)
    pickle.dump(dict_ccs, open(out_file + '.ccs_single_level.dict', 'wb'), -1)
    print(max_visit_len, max_seqs_len, len(dict_ccs), len(dict_3digit))


if __name__ == '__main__':
    dir_path = '../../../'

    admissionFile = dir_path + '../../../medicalAI_V2/dataset/mimic3/ADMISSIONS.csv'
    diagnosisFile = dir_path+ '../../../medicalAI_V2/dataset/mimic3/DIAGNOSES_ICD.csv'
    single_file = dir_path + 'ccs/ccs_single_dx_tool_2015.csv'
    multi_file = dir_path + 'ccs/ccs_multi_dx_tool_2015.csv'
    rd_out_file = dir_path + 'outputs/kemce/data/readmission_diagnosis/mimic'
    ml_out_file = dir_path + 'outputs/kemce/data/mortality_los/mimic'

    mimic_processing_for_readm_dx(admissionFile, diagnosisFile, single_file, multi_file, rd_out_file)
    # mimic_processing_for_mort_los(admissionFile, diagnosisFile, ml_out_file)

