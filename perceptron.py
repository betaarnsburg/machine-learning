# dataset: Minist

import numpy as np
import time

def loadData(fileName):
    '''
    load Mnist dataset
    :param fileName:loading path
    :return: dataset and label in list format
    '''
    print('start to read data')
    dataArr = []; labelArr = []
    fr = open(fileName, 'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        if int(curLine[0]) >= 5:
            labelArr.append(1)
        else:
            labelArr.append(-1)
        dataArr.append([int(num)/255 for num in curLine[1:]])
    return dataArr, labelArr

def perceptron(dataArr, labelArr, iter=50):
    '''
    training
    :param dataArr:train dataset (list)
    :param labelArr: train label(list)
    :param iter: iterations default=50
    :return: weights and bias
    '''
    print('start to trans')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m, n = np.shape(dataMat)
    w = np.zeros((1, np.shape(dataMat)[1]))
    b = 0
    h = 0.0001

    for k in range(iter):
        for i in range(m):
            xi = dataMat[i]
            yi = labelMat[i]
            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + h *  yi * xi
                b = b + h * yi
        print('Round %d:%d training' % (k, iter))
    return w, b


def model_test(dataArr, labelArr, w, b):
    '''
    test the acc
    :param dataArr:test dataset
    :param labelArr: test label
    :param w: weights
    :param b: bias
    :return: acc
    '''
    print('start to test')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T

    m, n = np.shape(dataMat)
    errorCnt = 0
    for i in range(m):
        xi = dataMat[i]
        yi = labelMat[i]
        result = -1 * yi * (w * xi.T + b)
        if result >= 0: errorCnt += 1
    accruRate = 1 - (errorCnt / m)
    return accruRate

if __name__ == '__main__':
    start = time.time()

    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    testData, testLabel = loadData('../Mnist/mnist_test.csv')

    w, b = perceptron(trainData, trainLabel, iter = 30)
    accruRate = model_test(testData, testLabel, w, b)

    end = time.time()
    print('accuracy rate is:', accruRate)
    print('time span:', end - start)
