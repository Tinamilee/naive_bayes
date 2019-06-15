from numpy import *
import matplotlib.pyplot as plt

def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels
# datMat, classLabels = loadSimpData()
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

# Decision stump–generating functions
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                # print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" % (i, threshVal, \
                #                                                                 inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst
D = mat(ones((5, 1))/5)
# buildStump(datMat, classLabels, D)

# AdaBoost training with decision stumps
# AdaBoost: adaptive boosting
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    # It holds the weight of each piece of data.
    # Initially, you’ll set  all  of  these  values  equal.
    # On  subsequent  iterations,
    # the  AdaBoost  algorithm  will increase the weight of the misclassified pieces of data
    # and decrease the weight of the properly classified data.
    # D is a probability distribution, so the sum of all the elements in D must be 1.0.
    # To meet this requirement, you initialize every element to 1/m.
    D = mat(ones((m, 1))/m)
    # column vector,aggClassEst，gives you the aggregate estimate of the class for every data point.
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        # This "buildStump" function takes D, the weights vector, and returns the stump with the lowest error using D.
        # The lowest error value is also returned as well as a vector with the estimated classes for this iteration D.
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        # Next, alpha is calculated.This will tell the total classifier how much to weight the output from this stump.
        # The statement max(error,1e-16)
        # is there to make sure you don’t have a divide-by-zero error in the case where there’s no error
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        #  The alpha value is added to the bestStump dictionary, and the dictionary is appended to the list.
        # This dictionary will contain all you need for classification.
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        # The next three lines are used to calculate new weights D for the next iteration.
        # In the case that you have 0 training error, you want to exit the for loop early.
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        # This is calculated by keeping a running sum of the estimated class in aggClassEst !!!
        # This value is a floating point number, and to get the binary class you use the sign() function.
        # If the total error is 0, you quit the for loop with the break statement.
        aggClassEst += alpha*classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    # return weakClassArr
    return weakClassArr, aggClassEst
datArr, labelArr = loadDataSet('horseColicTraining2.txt')
# classifierArray = adaBoostTrainDS(datArr, labelArr, 10)
classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)

# print(classifierArray)
# AdaBoost classification function
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        # his class estimate is multiplied by the alpha value for each stump and added to the total: aggClassEst
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print(aggClassEst)
    return sign(aggClassEst)

testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
prediction10 = adaClassify(testArr, classifierArray)
# print(prediction10)
# print(adaClassify([[5, 5], [0, 0]], classifierArray))
# Adaptive load data function
errArr = mat(ones((67, 1)))
# print(errArr[prediction10 != mat(testLabelArr).T].sum())

# ROC plotting and AUC calculating function
# predStrengths: the first is a NumPy array or matrix in a row
# vector form. This is the strength of the classifier’s predictions. Our classifier and our
# training functions generate this before they apply it to the sign() function.
def plotROC(predStrengths, classLabels):
    # This holds your cursor for plotting 游标
    cur = (1.0, 1.0)
    # calculating the AUC (area under the curve).
    ySum = 0.0
    # numPosClas: the number of positive instances you have by using array filtering
    # This will give you the number of steps you’re going to take in the y direction.
    # You’re going to plot in the range of 0.0 to 1.0 on both the x- and y-axes,
    # so to get the y step size you take 1.0/numPosClas. You can similarly get the x step size.
    numPosClas = sum(array(classLabels) == 1.0)
    # y: True Positive Rate
    yStep = 1/float(numPosClas)
    # x: False Positive Rate
    xStep = 1/float(len(classLabels) - numPosClas)
    # get sorted index
    # You next get the sorted index, but it’s from smallest to largest, so you start at the
    # point 1.0,1.0 and draw to 0,0
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # As you’re going through the list,
    for index in sortedIndicies.tolist()[0]:
        # you take a step down in the y direction every time you get a class of 1.0,
        # which decreases the true positive rate.
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        # # Similarly, you take a step backward in the x direction (false positive rate) for every other class.
        else:
            delX = xStep
            delY = 0
            # To compute the AUC, you need to add up a bunch of small rectangles
            # The width of each of these rectangles will be xStep, so you can add the heights of all the rectangles
            # and multiply the sum of the heights by xStep once to get the total area.( ySum*xStep
            # # The height sum (ySum) increases every time you move in the x direction.
            ySum += cur[1]
        # Once you’ve decided whether you’re going to move in the x or y direction, you draw a small,
        # straight-line segment from the current point to the new point. The current point, cur,
        # is then updated. Finally, you make the plot look nice and display it by printing the
        # AUC to the terminal.
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)
plotROC(aggClassEst.T, labelArr)




















