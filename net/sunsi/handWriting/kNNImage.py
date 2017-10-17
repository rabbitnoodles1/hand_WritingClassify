# coding=utf-8
from os import listdir
from numpy import *
import operator

'''
k-近邻分类算法
'''


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDistIndex = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


'''
将图像转化为测试向量
'''


def img2vector(filename):
    returnVector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, 32 * i + j] = int(lineStr[j])
    return returnVector


'''
训练算法
'''


def handWritingClassifyTrain():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    return trainingMat, hwLabels


'''
构建可用的系统
'''


def handWritingClassify():
    fileName = raw_input('输入手写数字的txt文件名（例如：0_0.txt），从testDigits文件夹中挑选：')
    hwVector = img2vector('testDigits/%s' % fileName)
    trainingMat, hwLabels = handWritingClassifyTrain()
    classifyResult = classify0(hwVector, trainingMat, hwLabels, 3)
    print "你手写的数字是： ", classifyResult