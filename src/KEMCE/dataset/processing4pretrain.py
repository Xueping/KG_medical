from datetime import datetime
import pickle


def mimic_processing(adm_file, dx_file, out_file):

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

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        new_admIdList = []
        for admId in admIdList:
            if admId in admDxMap:
                new_admIdList.append(admId)
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in new_admIdList])
        if len(sortedList) == 0: continue
        pidSeqMap[pid] = sortedList

    print('Building pids, dates, strSeqs')
    seqs = []
    vocab_set = {}
    for pid, visits in pidSeqMap.items():
        sep = []
        for i, visit in enumerate(visits):
            visit_set = visit[1]
            sep.append(visit_set)
            for code in visit_set:
                if code in vocab_set:
                    vocab_set[code] += 1
                else:
                    vocab_set[code] = 1
        seqs.append(sep)
    sorted_vocab = {k: v for k, v in sorted(vocab_set.items(), key=lambda item: item[1], reverse=True)}

    pickle.dump(seqs, open(out_file+'_pre_train.seqs', 'wb'), -1)
    print('number of valid patients: {}'.format(len(seqs)))

    outfd = open(out_file + '_pre_train_vocab.txt', 'w')
    for k, v in sorted_vocab.items():
        outfd.write(k + '\n')
    outfd.close()


if __name__ == '__main__':
    dir_path = '../../../'

    admissionFile = dir_path + '../../../medicalAI_V2/dataset/mimic3/ADMISSIONS.csv'
    diagnosisFile = dir_path+ '../../../medicalAI_V2/dataset/mimic3/DIAGNOSES_ICD.csv'
    outFile = dir_path + 'outputs/kemce/data/raw/mimic'
    mimic_processing(admissionFile, diagnosisFile, outFile)

