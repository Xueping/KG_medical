import pickle
import collections


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    vocab['PAD'] = len(vocab)
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = len(vocab)
    return vocab


def mimic_tree_building(seqs_file, types_file, graph_file, out_file):

    print('Read Saved data dictionary')
    types = load_vocab(types_file)
    seqs = pickle.load(open(seqs_file, 'rb'))

    startSet = set(types.keys())
    hitList = []
    cat1count = 0
    cat2count = 0
    cat3count = 0
    cat4count = 0

    infd = open(graph_file, 'r')
    _ = infd.readline()

    # add ancestors to dictionary
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

        if icd9 not in types:
            continue
        else:
            hitList.append(icd9)

        if desc1 not in types:
            cat1count += 1
            types[desc1] = len(types)

        if len(cat2) > 0:
            if desc2 not in types:
                cat2count += 1
                types[desc2] = len(types)
        if len(cat3) > 0:
            if desc3 not in types:
                cat3count += 1
                types[desc3] = len(types)
        if len(cat4) > 0:
            if desc4 not in types:
                cat4count += 1
                types[desc4] = len(types)
    infd.close()

    # add root_code
    types['A_ROOT'] = len(types)
    rootCode = types['A_ROOT']

    missSet = startSet - set(hitList)
    missSet.remove('PAD')  # comment this line for GRAM and KAME, work for KEMCE
    print('missing code: {}'.format(len(missSet)))
    print(cat1count,cat2count,cat3count, cat4count )

    fiveMap = {}
    fourMap = {}
    threeMap = {}
    twoMap = {}
    oneMap = dict([(types[icd], [types[icd], rootCode]) for icd in missSet])

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

        if icd9 not in types: continue

        icdCode = types[icd9]

        if len(cat4) > 0:
            code4 = types[desc4]
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fiveMap[icdCode] = [icdCode, rootCode, code1, code2, code3, code4]


        elif len(cat3) > 0:
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fourMap[icdCode] = [icdCode, rootCode, code1, code2, code3]

        elif len(cat2) > 0:
            code2 = types[desc2]
            code1 = types[desc1]
            threeMap[icdCode] = [icdCode, rootCode, code1, code2]

        else:
            code1 = types[desc1]
            twoMap[icdCode] = [icdCode, rootCode, code1]

    # Now we re-map the integers to all medical leaf codes.
    newFiveMap = collections.OrderedDict()
    newFourMap = collections.OrderedDict()
    newThreeMap = collections.OrderedDict()
    newTwoMap = collections.OrderedDict()
    newOneMap = collections.OrderedDict()
    newTypes = collections.OrderedDict()
    rtypes = dict([(v, k) for k, v in types.items()])

    codeCount = 0
    newTypes['PAD'] = codeCount # comment this line for GRAM and KAME, work for KEMCE
    codeCount += 1 # comment this line for GRAM and KAME, work for KEMCE
    for icdCode, ancestors in fiveMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newFiveMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1

    for icdCode, ancestors in fourMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newFourMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1

    for icdCode, ancestors in threeMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newThreeMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1

    for icdCode, ancestors in twoMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newTwoMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1

    for icdCode, ancestors in oneMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newOneMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1

    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                newVisit.append(newTypes[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    pickle.dump(newFiveMap, open(outFile + '.level5.pk', 'wb'), -1)
    pickle.dump(newFourMap, open(outFile + '.level4.pk', 'wb'), -1)
    pickle.dump(newThreeMap, open(outFile + '.level3.pk', 'wb'), -1)
    pickle.dump(newTwoMap, open(outFile + '.level2.pk', 'wb'), -1)
    pickle.dump(newOneMap, open(outFile + '.level1.pk', 'wb'), -1)
    pickle.dump(newTypes, open(outFile + '.types', 'wb'), -1)
    pickle.dump(newSeqs, open(outFile + '.seqs', 'wb'), -1)
    print(len(newTypes),len(newSeqs))


if __name__ == '__main__':

    dir_path = '../../../'
    infile = dir_path + 'ccs/ccs_multi_dx_tool_2015.csv'
    # for MIMIC III dataset
    # seqFile = dir_path + 'outputs/kemce/data/seq_prediction/mimic.inputs_all.seqs'
    # typeFile = dir_path + 'outputs/kemce/data/seq_prediction/mimic.vocab.txt'
    # outFile = dir_path + 'outputs/kemce/data/seq_prediction/mimic'

    #for eICU dataset
    seqFile = dir_path + 'outputs/eICU/seq_prediction/eicu.inputs_all.seqs'
    typeFile = dir_path + 'outputs/eICU/seq_prediction/eicu.vocab.txt'
    outFile = dir_path + 'outputs/eICU/seq_prediction/eicu'

    mimic_tree_building(seqFile, typeFile, infile, outFile)
