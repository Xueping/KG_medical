from datetime import datetime
import pickle
from GRAM.gram_helpers import convert_to_icd9, convert_to_3digit_icd9
from KEMCE.dataset import LabelsForData
import pandas as pd
import argparse


def mimic_processing(adm_file, dx_file, multi_dx_file, single_dx_file, out_file):

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
            # print('code without length: ', line)
            continue

        dxStr = 'D_' + convert_to_icd9(dx) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
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
    pids = []
    seqs = []
    seqs_span = []
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
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
    types_3digit = {}
    newSeqs_3digit = []
    for patient in seqs_3digit:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in set(visit):
                if code in types_3digit:
                    newVisit.append(types_3digit[code])
                else:
                    types_3digit[code] = len(types_3digit)
                    newVisit.append(types_3digit[code])
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

    print('Converting strSeqs to intSeqs, and making types')
    types = {}
    # types['PAD'] = len(types)  # comment this line for GRAM and KAME, work for KEMCE
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    pickle.dump(pids, open(out_file+'.pids', 'wb'), -1)
    pickle.dump(newSeqs, open(out_file+'.seqs', 'wb'), -1)
    pickle.dump(types, open(out_file+'.types', 'wb'), -1)
    pickle.dump(newSeqs_3digit, open(out_file+'.3digitICD9.seqs', 'wb'), -1)
    pickle.dump(types_3digit, open(out_file+'.3digitICD9.types', 'wb'), -1)
    pickle.dump(newSeqs_ccs, open(out_file + '.ccsSingleLevel.seqs', 'wb'), -1)
    pickle.dump(dict_ccs, open(out_file + '.ccsSingleLevel.types', 'wb'), -1)
    print(len(newSeqs),len(types), len(types_3digit), len(dict_ccs))


def eicu_processing(pats_file, dx_file, multi_dx_file, single_dx_file, out_file):
    dxes = pd.read_csv(dx_file, header=0)
    pats = pd.read_csv(pats_file, header=0)
    label4data = LabelsForData(multi_dx_file, single_dx_file)

    # unique patient:count
    pat_vc = pats.uniquepid.value_counts()

    # patients whose admission number is at least 2
    pat_two_plus = pat_vc[pat_vc > 1].index.tolist()

    # pid mapping admission list
    print('pid mapping admission list')
    pid_adm_map = {}
    for pid in pat_two_plus:
        pats_adm = pats[pats.uniquepid == pid]
        sorted_adms = pats_adm.sort_values(by=['hospitaldischargeyear', 'hospitaladmitoffset'],
                                           ascending=[True, False])['patientunitstayid'].tolist()
        pid_adm_map[pid] = sorted_adms

    # filter null in icc9code field
    dxes = dxes[dxes.icd9code.notnull()]

    # Building Building strSeqs
    print('Building Building strSeqs')
    seqs = []
    for pid, adms in pid_adm_map.items():
        seq = []
        for adm in adms:
            code_list = []
            diags = dxes[dxes.patientunitstayid == adm]
            for index, row in diags.iterrows():
                codes = row.icd9code.split(',')
                if len(codes) == 2:
                    # if the first letter is digit, it is icd9 code
                    if codes[0][0].isdigit():
                        code_list.append(codes[0].replace('.', ''))
                    if codes[1][0].isdigit():
                        code_list.append(codes[0].replace('.', ''))
                else:
                    if codes[0][0].isdigit():
                        code_list.append(codes[0].replace('.', ''))
            if len(code_list) > 0:
                seq.append(code_list)
        if len(seq) > 1:
            seqs.append(seq)

    # Building Building new strSeqs, which filters the admission with only one diagnosis code
    print('Building Building new strSeqs, which filters the admission with only one diagnosis code')
    new_seqs = []
    for seq in seqs:
        new_seq = []
        for adm in seq:
            if len(adm) == 1:
                continue
            else:
                code_set = set(adm)
                if len(code_set) == 1:
                    continue
                else:
                    new_seq.append(list(code_set))
        if len(new_seq) > 1:
            new_seqs.append(new_seq)

    # Building Building strSeqs, and string labels
    print('Building Building strSeqs, and string labels')
    new_seqs_str = []
    adm_dx_ccs = []
    adm_dx_ccs_cat1 = []
    for seq in new_seqs:
        seq_ls = []
        dx_ccs_ls = []
        dx_ccs_cat1_ls = []
        for adm in seq:
            new_adm = []
            dx_ccs = []
            dx_ccs_cat1 = []
            for dx in adm:
                dxStr = 'D_' + convert_to_icd9(dx)
                # dxStr = 'D_' + dx
                dxStr_ccs_single = 'D_' + label4data.code2single_dx[dx]
                dxStr_ccs_cat1 = 'D_' + label4data.code2first_level_dx[dx]
                new_adm.append(dxStr)
                dx_ccs.append(dxStr_ccs_single)
                dx_ccs_cat1.append(dxStr_ccs_cat1)
            seq_ls.append(new_adm)
            dx_ccs_ls.append(dx_ccs)
            dx_ccs_cat1_ls.append(dx_ccs_cat1)
        new_seqs_str.append(seq_ls)
        adm_dx_ccs.append(dx_ccs_ls)
        adm_dx_ccs_cat1.append(dx_ccs_cat1_ls)

    print('Converting strSeqs to intSeqs, and making types for ccs single-level code')
    dict_ccs = {}
    new_seqs_ccs = []
    for patient in adm_dx_ccs:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in set(visit):
                if code in dict_ccs:
                    new_visit.append(dict_ccs[code])
                else:
                    dict_ccs[code] = len(dict_ccs)
                    new_visit.append(dict_ccs[code])
            new_patient.append(new_visit)
        new_seqs_ccs.append(new_patient)

    print('Converting strSeqs to intSeqs, and making types for ccs multi-level first level code')
    dict_ccs_cat1 = {}
    new_seqs_ccs_cat1 = []
    for patient in adm_dx_ccs_cat1:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in set(visit):
                if code in dict_ccs_cat1:
                    new_visit.append(dict_ccs_cat1[code])
                else:
                    dict_ccs_cat1[code] = len(dict_ccs_cat1)
                    new_visit.append(dict_ccs_cat1[code])
            new_patient.append(new_visit)
        new_seqs_ccs_cat1.append(new_patient)

    print('Converting strSeqs to intSeqs, and making types')
    types = {}
    newSeqs = []
    for patient in new_seqs_str:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    pickle.dump(newSeqs, open(out_file + '.seqs', 'wb'), -1)
    pickle.dump(types, open(out_file + '.types', 'wb'), -1)
    pickle.dump(new_seqs_ccs_cat1, open(out_file + '.ccs_cat1.seqs', 'wb'), -1)
    pickle.dump(dict_ccs_cat1, open(out_file + '.ccs_cat1.types', 'wb'), -1)
    pickle.dump(new_seqs_ccs, open(out_file + '.ccsSingleLevel.seqs', 'wb'), -1)
    pickle.dump(dict_ccs, open(out_file + '.ccsSingleLevel.types', 'wb'), -1)
    print(len(newSeqs), len(types), len(dict_ccs_cat1), len(dict_ccs))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir.")
    args = parser.parse_args()

    # dir_path = '../../'
    dir_path = args.data_dir
    sigle_dx_file = dir_path + 'ccs/ccs_single_dx_tool_2015.csv'
    multi_dx_file = dir_path + 'ccs/ccs_multi_dx_tool_2015.csv'

    # data_source = 'mimic'
    data_source = 'eicu'

    if data_source == 'mimic':
        admissionFile = dir_path + 'data/ADMISSIONS.csv'
        diagnosisFile = dir_path + 'data/DIAGNOSES_ICD.csv'
        outFile = dir_path + 'outputs/gram/data/mimic/mimic'
        mimic_processing(admissionFile, diagnosisFile, multi_dx_file, sigle_dx_file, outFile)
    else:
        admissionFile = dir_path + 'data/patient.csv'
        diagnosisFile = dir_path + 'data/diagnosis.csv'
        outFile = dir_path + 'outputs/gram/data/eicu/eicu'
        eicu_processing(admissionFile, diagnosisFile, multi_dx_file, sigle_dx_file, outFile)

