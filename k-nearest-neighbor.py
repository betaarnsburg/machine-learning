# dataset: Minist
#k=25

import numpy as np
import time

def loadData(fileName):
    print('start read file')
    #restore the data
    dataArr = []; labelArr = []
    #read data
    fr = open(fileName)
    #travesal each line
    for line in fr.readlines():
        curLine = line.strip().split(',')
        dataArr.append([int(num) for num in curLine[1:]])
        labelArr.append(int(curLine[0]))
    #return dataset and label
    return dataArr, labelArr

def calcDist(x1, x2):
    '''
    compute the distance between two vectors
    L2
    :param x1:Vector1
    :param x2:Vector2
    :return:L2
    '''
    return np.sqrt(np.sum(np.square(x1 - x2)))

    #Manhattan distance
    #return np.sum(x1 - x2)




def getClosest(trainDataMat, trainLabelMat, x, topK):
    '''
    labeling

    :param trainDataMat:train dataset
    :param trainLabelMat:train lable
    :param x:predicting specimen x
    :param topK: K, very important. It is related to the accuracy
    :return:predicted label
    '''
    distList = [0] * len(trainLabelMat)
    for i in range(len(trainDataMat)):
        x1 = trainDataMat[i]
        curDist = calcDist(x1, x)
        distList[i] = curDist

    topKList = np.argsort(np.array(distList))[:topK]        #sorted
    labelList = [0] * 10
    for index in topKList:
        labelList[int(trainLabelMat[index])] += 1
    return labelList.index(max(labelList))


def model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK):
    '''
    test the acc
    :param trainDataArr:train dataset
    :param trainLabelArr: train label
    :param testDataArr: test dataset
    :param testLabelArr: test label
    :param topK: K value
    :return: acc
    '''
    print('start test')
    trainDataMat = np.mat(trainDataArr); trainLabelMat = np.mat(trainLabelArr).T
    testDataMat = np.mat(testDataArr); testLabelMat = np.mat(testLabelArr).T

    errorCnt = 0
    # for i in range(len(testDataMat)):
    for i in range(200):
        # print('test %d:%d'%(i, len(trainDataArr)))
        print('test %d:%d' % (i, 200))
        x = testDataMat[i]
        y = getClosest(trainDataMat, trainLabelMat, x, topK)
        if y != testLabelMat[i]: errorCnt += 1

    #return the acc
    # return 1 - (errorCnt / len(testDataMat))
    return 1 - (errorCnt / 200)



if __name__ == "__main__":
    start = time.time()

    #obtain the train dataset
    trainDataArr, trainLabelArr = loadData('../Mnist/mnist_train.csv')
    #obtain the test dataset
    testDataArr, testLabelArr = loadData('../Mnist/mnist_test.csv')
    #acc
    accur = model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 25)
    print('accur is:%d'%(accur * 100), '%')

    end = time.time()
    print('time span:', end - start)
