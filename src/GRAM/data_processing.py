from datetime import datetime
import pickle
from GRAM.gram_helpers import convert_to_icd9, convert_to_3digit_icd9
from KEMCE.dataset import LabelsForData


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


if __name__ == '__main__':
    dir_path = '../../'

    admissionFile = dir_path + '../../../medicalAI_V2/dataset/mimic3/ADMISSIONS.csv'
    diagnosisFile = dir_path+ '../../../medicalAI_V2/dataset/mimic3/DIAGNOSES_ICD.csv'
    # outFile = dir_path + 'outputs/kame/data/mimic'
    # outFile = dir_path + 'outputs/gram/data/mimic/mimic'
    outFile = dir_path + 'outputs/kemce/data/mimic'
    sigle_dx_file = dir_path + 'ccs/ccs_single_dx_tool_2015.csv'
    multi_dx_file = dir_path + 'ccs/ccs_multi_dx_tool_2015.csv'
    mimic_processing(admissionFile, diagnosisFile, multi_dx_file, sigle_dx_file, outFile)

