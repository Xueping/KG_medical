import pickle


def mimic_tree_building(seqs_file, types_file, graph_file, out_file):

    print('Read Saved Sequences, data dictionary')
    seqs = pickle.load(open(seqs_file, 'rb'))
    types = pickle.load(open(types_file, 'rb'))

    startSet = set(types.keys())
    hitList = []
    missList = []
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

        if icd9.startswith('E'):
            if len(icd9) > 4: icd9 = icd9[:4] + '.' + icd9[4:]
        else:
            if len(icd9) > 3: icd9 = icd9[:3] + '.' + icd9[3:]
        icd9 = 'D_' + icd9
        if icd9 not in types: continue

        if icd9 not in types:
            missList.append(icd9)
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

    fiveMap = {}
    fourMap = {}
    threeMap = {}
    twoMap = {}
    oneMap = dict([(types[icd], [types[icd], rootCode]) for icd in missSet])

    # to store internal nodes and corresponding ancestors
    anc_FourMap = {}
    anc_ThreeMap = {}
    anc_TwoMap = {}
    anc_OneMap = {}

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

        if icd9.startswith('E'):
            if len(icd9) > 4: icd9 = icd9[:4] + '.' + icd9[4:]
        else:
            if len(icd9) > 3: icd9 = icd9[:3] + '.' + icd9[3:]
        icd9 = 'D_' + icd9

        if icd9 not in types: continue
        icdCode = types[icd9]

        if len(cat4) > 0:
            code4 = types[desc4]
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fiveMap[icdCode] = [icdCode, rootCode, code1, code2, code3, code4]

            anc_FourMap[code4] = [code4, rootCode, code1, code2, code3]
            anc_ThreeMap[code3] = [code3, rootCode, code1, code2]
            anc_TwoMap[code2] = [code2, rootCode, code1]
            anc_OneMap[code1] = [code1, rootCode]

        elif len(cat3) > 0:
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fourMap[icdCode] = [icdCode, rootCode, code1, code2, code3]

            anc_ThreeMap[code3] = [code3, rootCode, code1, code2]
            anc_TwoMap[code2] = [code2, rootCode, code1]
            anc_OneMap[code1] = [code1, rootCode]
        elif len(cat2) > 0:
            code2 = types[desc2]
            code1 = types[desc1]
            threeMap[icdCode] = [icdCode, rootCode, code1, code2]

            anc_TwoMap[code2] = [code2, rootCode, code1]
            anc_OneMap[code1] = [code1, rootCode]

        else:
            code1 = types[desc1]
            twoMap[icdCode] = [icdCode, rootCode, code1]

            anc_OneMap[code1] = [code1, rootCode]
    anc_OneMap[rootCode] = [rootCode, rootCode]

    # Now we re-map the integers to all medical leaf codes.
    newFiveMap = {}
    newFourMap = {}
    newThreeMap = {}
    newTwoMap = {}
    newOneMap = {}
    newTypes = {}
    rtypes = dict([(v, k) for k, v in types.items()])

    codeCount = 0
    # newTypes['PAD'] = codeCount # comment this line for GRAM and KAME, work for KEMCE
    # codeCount += 1 # comment this line for GRAM and KAME, work for KEMCE
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
                newVisit.append(newTypes[rtypes[code]])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    pickle.dump(newFiveMap, open(outFile + '.level5.pk', 'wb'), -1)
    pickle.dump(newFourMap, open(outFile + '.level4.pk', 'wb'), -1)
    pickle.dump(newThreeMap, open(outFile + '.level3.pk', 'wb'), -1)
    pickle.dump(newTwoMap, open(outFile + '.level2.pk', 'wb'), -1)
    pickle.dump(newOneMap, open(outFile + '.level1.pk', 'wb'), -1)

    pickle.dump(anc_FourMap, open(outFile + '.a_level4.pk', 'wb'), -1)
    pickle.dump(anc_ThreeMap, open(outFile + '.a_level3.pk', 'wb'), -1)
    pickle.dump(anc_TwoMap, open(outFile + '.a_level2.pk', 'wb'), -1)
    pickle.dump(anc_OneMap, open(outFile + '.a_level1.pk', 'wb'), -1)

    pickle.dump(newTypes, open(outFile + '.types', 'wb'), -1)
    pickle.dump(newSeqs, open(outFile + '.seqs', 'wb'), -1)
    print(len(newTypes), len(newSeqs))


if __name__ == '__main__':

    dir_path = '../../'
    infile = dir_path + 'ccs/ccs_multi_dx_tool_2015.csv'
    # seqFile = dir_path + 'outputs/gram/data/mimic.seqs'
    # typeFile = dir_path + 'outputs/gram/data/mimic.types'
    # outFile = dir_path + 'outputs/gram/data/mimic'

    # seqFile = dir_path + 'outputs/kame/data/mimic.seqs'
    # typeFile = dir_path + 'outputs/kame/data/mimic.types'
    # outFile = dir_path + 'outputs/kame/data/mimic'

    seqFile = dir_path + 'outputs/kemce/data/mimic/mimic.seqs'
    typeFile = dir_path + 'outputs/kemce/data/mimic/mimic.types'
    outFile = dir_path + 'outputs/kemce/data/mimic/mimic'

    mimic_tree_building(seqFile, typeFile, infile, outFile)
