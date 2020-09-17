import numpy as np
from icd9cms.icd9 import search
import pickle


def ccs_kg_building(graph_file, out_file):
    _TEST_RATIO = 0.2
    _VALIDATION_RATIO = 0.1
    entities = {}
    code2desc = {}
    relation = {
        'is_parent_of': 0,
        'is_child_of': 1
    }
    train_set = set()

    infd = open(graph_file, 'r')
    _ = infd.readline()

    # build dictionary of entities
    for line in infd:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_L1_' + cat1
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_L2_' + cat2
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_L3_' + cat3
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_L4_' + cat4

        icd9 = 'D_' + icd9

        if icd9 not in entities:
            entities[icd9] = len(entities)

        if desc1 not in entities:
            entities[desc1] = len(entities)

        if len(cat2) > 0:
            if desc2 not in entities:
                entities[desc2] = len(entities)
        if len(cat3) > 0:
            if desc3 not in entities:
                entities[desc3] = len(entities)
        if len(cat4) > 0:
            if desc4 not in entities:
                entities[desc4] = len(entities)

        sr = search(icd9[2:])
        i = 1
        while sr is None:
            # print(sr)
            l = len(icd9)
            if l-i == 0:
                break
            sr = search(icd9[2:l-i])
            i += 1

        if sr is not None:
            ds = str(sr).split(':')
            if ds[2] == 'None':
                code2desc[icd9] = ds[1]
            else:
                code2desc[icd9] = ds[2]
    infd.close()

    # add root_code
    entities['A_ROOT'] = len(entities)
    rootCode = entities['A_ROOT']

    infd = open(infile, 'r')
    infd.readline()

    for line in infd:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_L1_' + cat1
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_L2_' + cat2
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_L3_' + cat3
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_L4_' + cat4

        icd9 = 'D_' + icd9

        # if icd9 not in entities:
        #     continue
        icdCode = entities[icd9]

        if len(cat4) > 0:
            code4 = entities[desc4]
            code3 = entities[desc3]
            code2 = entities[desc2]
            code1 = entities[desc1]
            train_set.add(str(rootCode) + ',' + str(code1) + ',0')
            train_set.add(str(code1) + ',' + str(rootCode) + ',1')
            train_set.add(str(code1) + ',' + str(code2) + ',0')
            train_set.add(str(code2) + ',' + str(code1) + ',1')
            train_set.add(str(code2) + ',' + str(code3) + ',0')
            train_set.add(str(code3) + ',' + str(code2) + ',1')
            train_set.add(str(code3) + ',' + str(code4) + ',0')
            train_set.add(str(code4) + ',' + str(code3) + ',1')
            train_set.add(str(code4) + ',' + str(icdCode) + ',0')
            train_set.add(str(icdCode) + ',' + str(code4) + ',1')

        elif len(cat3) > 0:
            code3 = entities[desc3]
            code2 = entities[desc2]
            code1 = entities[desc1]

            train_set.add(str(rootCode) + ',' + str(code1) + ',0')
            train_set.add(str(code1) + ',' + str(rootCode) + ',1')
            train_set.add(str(code1) + ',' + str(code2) + ',0')
            train_set.add(str(code2) + ',' + str(code1) + ',1')
            train_set.add(str(code2) + ',' + str(code3) + ',0')
            train_set.add(str(code3) + ',' + str(code2) + ',1')
            train_set.add(str(code3) + ',' + str(icdCode) + ',0')
            train_set.add(str(icdCode) + ',' + str(code3) + ',1')

        elif len(cat2) > 0:
            code2 = entities[desc2]
            code1 = entities[desc1]
            train_set.add(str(rootCode) + ',' + str(code1) + ',0')
            train_set.add(str(code1) + ',' + str(rootCode) + ',1')
            train_set.add(str(code1) + ',' + str(code2) + ',0')
            train_set.add(str(code2) + ',' + str(code1) + ',1')
            train_set.add(str(code2) + ',' + str(icdCode) + ',0')
            train_set.add(str(icdCode) + ',' + str(code2) + ',1')

        else:
            code1 = entities[desc1]

            train_set.add(str(rootCode) + ',' + str(code1) + ',0')
            train_set.add(str(code1) + ',' + str(rootCode) + ',1')
            train_set.add(str(code1) + ',' + str(icdCode)+ ',0')
            train_set.add(str(icdCode) + ',' + str(code1) + ',1')

    outfd = open(out_file + 'relation2id', 'w')
    for k, v in relation.items():
        outfd.write(str(v) + ',' + k + '\n')
    outfd.close()

    outfd = open(out_file + 'entity2id', 'w')
    for k, v in entities.items():
        outfd.write(str(v) + ',' + k + '\n')
    outfd.close()

    edges_ls = list(train_set)
    data_size = len(edges_ls)
    ind = np.random.permutation(data_size)
    nTest = int(_TEST_RATIO * data_size)
    nValid = int(_VALIDATION_RATIO * data_size)

    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest + nValid]
    train_indices = ind[nTest + nValid:]

    train = [edges_ls[i] for i in train_indices]
    valid = [edges_ls[i] for i in valid_indices]
    test = [edges_ls[i] for i in test_indices]

    outfd = open(out_file + 'train_file', 'w')
    for edge in train:
        outfd.write(edge + '\n')
    outfd.close()

    outfd = open(out_file + 'valid_file', 'w')
    for edge in valid:
        outfd.write(edge + '\n')
    outfd.close()

    outfd = open(out_file + 'test_file', 'w')
    for edge in test:
        outfd.write(edge + '\n')
    outfd.close()

    with open(out_file + 'code2desc.pickle', 'wb') as handle:
        pickle.dump(code2desc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(len(code2desc))


if __name__ == '__main__':

    dir_path = '../../'
    infile = dir_path + 'ccs/ccs_multi_dx_tool_2015.csv'
    outFile = dir_path + 'outputs/kemce/KG/'

    ccs_kg_building(infile, outFile)
