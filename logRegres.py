import math
import numpy
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1 + numpy.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose()
    m, n = numpy.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = numpy.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights) # not one number but a column vector with as many elements as you have data points
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
        # Qualitatively you can see we’re calculating the error between the actual class
        # and the predicted class and then moving in the direction of that error. ！！！！划重点
    return weights

dataArr, labelMat = loadDataSet()
# weights = gradAscent(dataArr, labelMat)
# print(x)

def plotBestFit(wei):
    # weights = wei.getA()  # 将matrix转化成array
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = numpy.array(dataMat)
    n = numpy.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = numpy.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    # I set 0 = w0x0+w1x1+w2x2 (x0was0) and solved for X2 in terms of X1 {(h = 1/(1+exp(-wx))  == 1/2 ?)}
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m, n = numpy.shape(dataMatrix)
    alpha = 0.01
    global weis
    weis = numpy.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weis))
        error = classLabels[i] - h
        weis = weis + alpha * error * dataMatrix[i]
    return weis

# ws = stocGradAscent0(numpy.array(dataArr), labelMat)
# plotBestFit(ws)
def stocGradAscent0_1(dataMatrix, classLabels, numIter=20):
    m, n = numpy.shape(dataMatrix)
    alpha = 0.01
    weights = numpy.ones(n)
    weiCha = numpy.zeros((numIter, 3))
    for j in range(numIter):
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
        weiCha[j] = weights
    x = numpy.arange(0, 20, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = weiCha[:, 0]
    ax.plot(x, y)
    ax.plot(x, weiCha[:, 1], c='red')
    ax.plot(x, weiCha[:, 2], c='green')
    plt.xlabel('x')
    plt.ylabel('weights第0列')
    plt.show()
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = numpy.shape(dataMatrix)
    weights = numpy.ones(n)
    for j in range(numIter):
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(numpy.random.uniform(0, len(dataMatrix)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(randIndex)
    return weights
# weights = stocGradAscent1(numpy.array(dataArr), labelMat)
# plotBestFit(weights)
# stocGradAscent0_1(numpy.array(dataArr), labelMat)

def stocGradAscent1_1(dataMatrix, classLabels, numIter=4000):
    m, n = numpy.shape(dataMatrix)
    weights = numpy.ones(n)
    weiCha = numpy.zeros((numIter, 3))
    for j in range(numIter):
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(numpy.random.uniform(0, len(dataMatrix)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(randIndex)
        weiCha[j] = weights
    x = numpy.arange(0, 4000, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = weiCha[:, 0]
    ax.plot(x, y)
    ax.plot(x, weiCha[:, 1], c='red')
    ax.plot(x, weiCha[:, 2], c='green')
    plt.xlabel('x')
    plt.ylabel('weights第0列')
    plt.show()
    return weights
# stocGradAscent1_1(numpy.array(dataArr), labelMat)

# apply to real_word
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(numpy.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(numpy.array(lineArr), trainWeights)) != int(lineArr[20]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
multiTest()












