import numpy
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while (j==i):
        j = int(numpy.random.uniform(0, m))
    return j
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

dataArr, labelArr = loadDataSet('testSet6.txt')

#  a constant C
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose()
    b = 0
    m, n = numpy.shape(dataMatrix)
    alphas = numpy.mat(numpy.zeros((m, 1)))
    # This variable will hold a
    # count  of  the  number  of  times  you’ve  gone  through  the  dataset  without  any  alphas
    # changing.
    iter = 0
    while (iter < maxIter):
        #  alphaPairsChanged is  used  to  record  if  the  attempt  to optimize any alphas worked
        alphaPairsChanged = 0
        for i in range(m):
            # this is our prediction of the class
            fXi = float(numpy.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            # The error Ei is next calculated based on the prediction and the real class of this instance.
            Ei = fXi - float(labelMat[i])
            # If this error is large, then the alpha corresponding to this data instance can be optimized.
            # In the if statement, both the positive and negative margins are tested.
            # In this if statement, you also check to see that the alpha isn’t equal to 0 or C.
            # Alphas will be clipped at 0 or C, so if they’re equal to these, they’re “bound”
            # and can’t be increased or decreased, so it’s not worth trying to optimize these alphas.
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                # You calculate the“error for this alpha similar to what you did for the first alpha,alpha[i]
                fXj = float(numpy.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # so that later you can compare the new alphas and the old ones.
                # Python  passes  all  lists  by  reference,  so  you  have  to
                # explicitly  tell  Python  to  give  you  a  new  memory  location  for variable
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #  clamping alpha[j] between  0  and C
                #
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                #  ，L会等于？H？直接跳过？ otherwise bound
                #  can’t be increased or decreased, so
                # it’s not worth trying to optimize these alphas.
                if L == H:
                    print("L == H")
                    continue
                # Eta is the optimal amount to change alpha[j]. This is calculated in the long line of algebra.
                # If eta is 0, you also quit the current iteration of the for loop.
                # This step is a simplification of the real SMO algorithm.
                # if eta is 0, there’s a messy way to calculate the new  alpha[j],but we won’t get into that here.
                # You can read Platt’s original paper if you really want to know how that works.
                # It turns out this seldom occurs, so it’s  OK if you skip it
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                # calculate a new alpha[j] and clip it using the helper function
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # j not moving , continue
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # alpha[i] is changed by the same amount as alpha[j] but in the opposite  direction
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # After  you  optimize alpha[i] and alpha[j], you  set the constant term b for these two alphas
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T-labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                # reached the bottom of the for loop without hitting a continue statement,
                # then you’ve successfully changed a pair of alphas and you can increment alphaPairsChanged.
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        # Outside the for loop,  you  check  to  see  if  any alphas have been updated;
        # if so you set iter to 0 and continue.
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
        # you’ll only stop and exit the while loop
        # when you’ve gone through the entire dataset maxIter number of times without anything changing.
    return b, alphas
# b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = dataArr
# y = numpy.multiply(alphas*numpy.mat(dataArr).T) + b
# x = dataArr
# y = -alphas*x - b
# ax.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# print(b)
# print(alphas[alphas > 0])
# print(numpy.shape(alphas[alphas > 0]))
# for i in range(100):
#     if alphas[i] > 0.0:
#         print(dataArr[i], labelArr[i])

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = numpy.shape(dataMatIn)[0]
        self.alphas = numpy.mat(numpy.zeros((self.m, 1)))
        self.b = 0
        self.eCache = numpy.mat(numpy.zeros((self.m, 2)))
        # self.K为对称矩阵，k(xy)表示第x行数据和第y行数据的radial bias?
        # 由Gaussian version得到，k(x, y) = exp(-(x-y)**2/2*nita**2)
        self.K = numpy.mat(numpy.zeros((self.m, self.m)))
        # 循环m行数据，针对i行数据计算与另外其他行数据的radial bias，
        # kernelTrans计算第i行数据和另外每行数据的radial bias，并存入到self.K的第i列中
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)
def calcEk(oS, k):
    # fXk = float(numpy.multiply(oS.alphas, oS.labelMat).T*(oS.X*oS.X[k, :].T)) + oS.b
    # Ek = fXk - float(oS.labelMat[k])
    # use kernel
    fXk = float(numpy.multiply(oS.alphas, oS.labelMat).T*oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEchcheList = numpy.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEchcheList)) > 1:
        for k in validEchcheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        # eta = 2.0 * oS.X[i, :]*oS.X[j, :].T - oS.X[i, :]*oS.X[i, :].T - oS.X[j, :]*oS.X[j, :].T
        # use kernel
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        # b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i]-alphaIold)*oS.X[i, :]*oS.X[i, :].T\
        #      -oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i, :]*oS.X[j, :].T
        # b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T \
        #      - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        # use kernel
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*oS.K[i, i]\
                       - oS.labelMat[j] * (oS.alphas[j] - alphaJold)*oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold)*oS.K[i, j]\
                       - oS.labelMat[j] * (oS.alphas[j] - alphaJold)*oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        return 1
    else:
        return 0
#  the  constant  C  gives weight to different parts of the optimization problem !!!
#  C controls the balance between making sure all of the examples have a margin of at least 1.0
#  and making the margin as wide as possible.
# in this function an iteration is defined as one pass through the loop  regardless  of  what  was  done.
# it will stop if there are any oscillations in the optimization.
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup):
    oS = optStruct(numpy.mat(dataMatIn), numpy.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    #  You’ll exit from the loop whenever the number of iterations  exceeds  your  specified  maximum
    # or  you  pass  through  the  entire  set  without changing any alpha pairs.
    # maxIter in that function you counted an iteration as a pass through the entire set when no alphas were changed.
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = numpy.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # 循环过一遍后entireSet会变成False，满足第一个条件，程序跳出if_elif语句块，但此时alphaPairsChanged > 0
        # 即使满足第二个判断语句 alphaPairsChanged == 0， 但不会执行。即entireSet是False
        # 循环第二遍entireSet等于false，不满足第一个if语句。alphaPairsChanged == 0，满足第二个if语句entireSet==true，
        # 第三遍 entireSet等于True，满足第一个语句块，entireSet==false，程序跳出if_elif语句块，
        # 循环第四遍，alphaPairsChanged == 0,没有改变alpha， 而且又循环过一遍entireSet，不满足继续循环条件，退出
        # 每次循环伊始，alphaPairsChanged == 0，用来判断此次循环是否有alpha的改变
        # 第一个循环，如果entireSet过了一遍，有alphaPairsChanged改变（entireSet此时会是False），
        # 第二个循环，在false状态下， 过一遍nonBoundIs， alphaPairsChanged ！= 0（有所改变），entireSet就仍是保持False状态
        # 第三个循环，在entireSet仍是False的状态下（过了一遍entireSet且false状态下alphaPairsChanged 有所改变，
        # 此时如果没有alphaPairsChanged改变，循环将退出，不用再循环了
        # 跳出情况2：本来entireSet的True，循环一遍但没改变alphaPairsChanged，entireSet变成false，alphaPairsChanged为0
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
        # print("entireSet: %d" % entireSet)
        # print("alphaPairsChanged number: %d" % alphaPairsChanged)
    return oS.b, oS.alphas
# b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
# print(b)
# print(alphas[alphas > 0])
# print(numpy.shape(alphas[alphas > 0]))
# for i in range(100):
#     if alphas[i] > 0.0:
#         print(dataArr[i], labelArr[i])
# print(alphas > 0)
def calcWs(alphas, dataArr, classLabels):
    X = numpy.mat(dataArr)
    labelMat = numpy.mat(classLabels).transpose()
    m, n = numpy.shape(X)
    w = numpy.zeros((n, 1))
    for i in range(m):
        w += numpy.multiply(alphas[i]*labelMat[i], X[i, :].T)
    return w
# ws = calcWs(alphas, dataArr, labelArr)
# dataMat = numpy.mat(dataArr)
# labelArr[0] = -1
# e0 = dataMat[0]*numpy.mat(ws)+b
# print(e0)
# ans2 = dataMat[2]*numpy.mat(ws)+b
# print(ans2)
# print("label2", labelArr[2])
# ans1 = dataMat[1]*numpy.mat(ws)+b
# print(ans1)
# print("label1", labelArr[1])
def kernelTrans(X, A, kTup):
    m, n = numpy.shape(X)
    K = numpy.mat(numpy.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow*deltaRow.T
        K = numpy.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K
# Radial bias test function for classifying with a kernel
def testRbf(k1 = 1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = numpy.mat(dataArr)
    labelMat = numpy.mat(labelArr).transpose()
    svInd = numpy.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % numpy.shape(sVs)[0])
    m, n = numpy.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T*numpy.multiply(labelSV, alphas[svInd]) + b
        if numpy.sign(predict) != numpy.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))
# testRbf()
# Support vector machine handwriting recognition
def img2vector(filename):
    returnVec = numpy.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32*i+j] = int(lineStr[j])
    return returnVec
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = numpy.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels
def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = numpy.mat(dataArr)
    labelMat = numpy.mat(labelArr).transpose()
    svInd = numpy.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % numpy.shape(sVs)[0])
    m, n = numpy.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T*numpy.multiply(labelSV, alphas[svInd]) + b
        if numpy.sign(predict) != numpy.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = numpy.mat(dataArr)
    labelMat = numpy.mat(labelArr).transpose()
    m, n = numpy.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * numpy.multiply(labelSV, alphas[svInd]) + b
        if numpy.sign(predict) != numpy.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))
testDigits(('rbf', 20))
























