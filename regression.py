from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    # to compute the determinate.
    if linalg.det(xTx) == 0:
        print("This matrix is singular, cannot do inverse")
        return
    #  as ws = linalg.solve(xTx,xMat.T*yMatT)
    ws = xTx.I * (xMat.T*yMat)
    return ws
# xArr, yArr = loadDataSet('ex0.txt')
# print(xArr[0:2])
# ws = standRegres(xArr, yArr)
# print(ws)
# xMat = mat(xArr)
# yMat = mat(yArr)
# yHat = xMat*ws
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
#
# xCopy = xMat.copy()
# xCopy.sort(0)
# yHat = xCopy*ws
# ax.plot(xCopy[:, 1], yHat)
# plt.show()
# co = corrcoef(yHat.T, yMat)
# print(co)
# Locally weighted linear regression function
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws
def lwlrTest(testArr ,xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat
# print(yArr[0])
# print(lwlr(xArr[0], xArr, yArr, 1.0))
# print(lwlr(xArr[0], xArr, yArr, 0.001))
# yHat = lwlrTest(xArr, xArr, yArr, 1)
# xMat = mat(xArr)
# srtInd = xMat[:, 1].argsort(0)
# xSort = xMat[srtInd][:, 0, :]
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:, 1], yHat[srtInd])
# ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
# plt.show()
def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()
abX, abY=loadDataSet('abalone.txt')
# yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
# yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
# yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)

# print(rssError(abY[100:199], yHat01.T))
# print(rssError(abY[100:199], yHat1.T))
# print(rssError(abY[100:199], yHat10.T))
# ws = standRegres(abX[0:99], abY[0:99])
# yHat = mat(abX[100:199])*ws
# yh = rssError(abY[100:199], yHat.T.A)
# print(yh)

# Ridge regression
# Ridge regression is a regression method that
# allows  you  to  compute  regression  coefficients
# despite  being  unable  to  compute  the inverse of XTX
#  Ridge regression is an example of a shrinkage method.
# Shrinkage methods impose a constraint on the size of the regression coefficient
def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i, :] = ws.T
    return wMat
# ridgeWeights = ridgeTest(abX, abY)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ridgeWeights)
# plt.show()

# regularize by columns
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)   #calc mean then subtract it off
    inVar = var(inMat, 0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

# Forward stagewise linear regression
# One is eps, the step size to take at each iteration,
# and the second is numIt, which is the number of iterations.
def stageWise(xArr, yArr, eps = 0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    weights = mat(eye((m)))
    # returnMat = mat((numIt, xMat.shape()[1]))
    returnMat = zeros((numIt, n))  # testing code remove
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat
xArr, yArr = loadDataSet('abalone.txt')
# print(stageWise(xArr, yArr, 0.001, 5000))
# stageWise：One thing to notice is that w1 and w6 are exactly 0.
# This means they don’t contribute anything to the result.
# These variables are probably not needed.
# With the eps variable set to 0.01,
# after some time the coefficients will all saturate and oscillate
# between certain values because the step size is too large.
#  ----------------------------
# xMat = mat(xArr)
# yMat = mat(yArr).T
# xMat = regularize(xMat)
# yM = mean(yMat, 0)
# yMat = yMat - yM
# weights = standRegres(xMat, yMat.T)
# print(weights.T)
#  ----------------------------
# swws = stageWise(xArr, yArr, 0.005, 1000)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(swws)
# plt.show()

# Shopping information retrieval function
from time import sleep
import json
import urllib
import socket
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    # 睡一会儿再取数据
    sleep(100)
    myAPIstr = 'get from code.google.com'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json'\
                % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print("problem with item %d" % i)
    # 请求太频繁不让取数据，close关闭会好点？
    pg.close()

def setDataCollect(retX, retY):
    #  设置socket默认的等待时间，在read超时后能自动往下继续跑
    socket.setdefaulttimeout(20)
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
lgX = []
lgY = []
setDataCollect(lgX, lgY)









