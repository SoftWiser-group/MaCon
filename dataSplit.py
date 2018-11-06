"""
1. Load data from Instagram.data;
2. Randomly sample certain number of historical posts for every user;
3. Split the whole set into train set and test set.
"""
import numpy as np
import random
import pickle
import math
random.seed(1)

numHist = 2


def doSplit():
    iFile = open("Instagram.data", "r")
    wholeSet = []
    count = 0
    for line in iFile.readlines():
        lineVec = line.strip().split("\t")
        tmp_text = [int(x) for x in lineVec[1].split(" ")]
        tmp_tag = [int(x) for x in lineVec[2].split(" ")]
        wholeSet.append((np.array(tmp_text), np.array(tmp_tag), lineVec[0], int(lineVec[3])))
        count += 1
        if count %20000 == 0:
            print(count)
            # break

    print("Loading data Done.")

    postDict = {}
    for inst in wholeSet:
        postDict[inst[2]] = inst
    nameId = {}
    for inst in wholeSet:
        if inst[-1] not in nameId.keys():
            nameId[inst[-1]] = []
        nameId[inst[-1]].append(inst[-2])
    user_samples = {}
    for name in nameId.keys():
        samples = random.sample(nameId[name], numHist)
        user_samples[name] = [postDict[i] for i in samples]
        nameId[name] = [i for i in nameId[name] if i not in samples]
    fileName = "user_post_samples_%d.pkl" % numHist
    pickle.dump(user_samples, open(fileName, "wb"))
    print("Getting samples Done...")

    trainId = []
    testId = []
    for name in nameId.keys():
        samples = random.sample(nameId[name], int(math.ceil(len(nameId[name]) / 10)))
        testId.extend(samples)
        trainId.extend([i for i in nameId[name] if i not in samples])
    sampleId = []
    for name in user_samples.keys():
        sampleId.extend(user_samples[name])
    print(len(trainId), len(testId), len(sampleId), len(wholeSet))

    trainInst = [postDict[i] for i in trainId]
    testInst = [postDict[i] for i in testId]
    print(len(trainId), len(testId))

    text_train = []
    tags_train = []
    ids_train = []
    user_train = []
    random.shuffle(trainInst)
    for inst in trainInst:
        text_train.append(inst[0])
        tags_train.append(inst[1])
        ids_train.append(inst[2])
        user_train.append(inst[3])
    pickle.dump((np.array(text_train), np.array(tags_train), ids_train, np.array(user_train)),
                open("trainSet_20_%d.pkl" % numHist, "wb"))

    text_test = []
    tags_test = []
    ids_test = []
    user_test = []
    random.shuffle(testInst)
    for inst in testInst:
        text_test.append(inst[0])
        tags_test.append(inst[1])
        ids_test.append(inst[2])
        user_test.append(inst[3])
    pickle.dump((np.array(text_test), np.array(tags_test), ids_test, np.array(user_test)),
                open("testSet_20_%d.pkl" % numHist, "wb"))
    # print("test set has %d instances." % len(user_test))
    print("Data set splitting Done.")


if __name__ == "__main__":
    doSplit()