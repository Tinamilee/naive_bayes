from numpy import *

class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

# CART tree-building code
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # !!!list(map(float, curLine))  not map(float, curLine)
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

# This takes three arguments: a dataset, a feature on which to split, and a value for that feature.
# The function returns two sets. The two sets are created using array filtering for the given feature and value.
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

#Regression tree split function
# regLeaf() generates the model for a leaf node.
# When chooseBestSplit() decides that you no longer should split the data,
# it will call regLeaf() to get a model for the leaf.
# The model in a regression tree is the mean value of the target variables.
def regLeaf(dataSet):
    return mean(dataSet[:, -1])
# error estimate, regErr().
# This function returns the squared error of the target variables in a given dataset.!!!??
# You could have first calculated the mean, then calculated the deviation, and then squared it,
# but it’s easier to call var(), which calculates the mean squared error
# You want the total squared error, not the mean,
# so you can get it by multiplying by the number of instances in a dataset.
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # These two values are user-defined settings that tell the function when to quit creating new splits.
    # The variable tolS is a tolerance on the error reduction,
    # and tolN is the minimum data instances to include in a split
    tolS = ops[0]
    tolN = ops[1]
    # check the number of unique values by creating a set from all the target variables.
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    # This error S will be checked against new values of the error to see if splitting reduces the error
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # exit if low error reduction
    # If splitting the dataset improves the error by only a small amount,
    # you choose not to split and create a leaf node.
    if (S-bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # exit if split creates small dataset
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue
# There are four arguments to createTree(): a dataset on which to build the tree and three optional arguments.
# The three optional arguments tell the function which type of tree to create.
# The argument leafType is the function used to create a leaf.
# The argument errType is a function used for measuring the error on the dataset.
# The last argument, ops, is a tuple of parameters for creating a tree.
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # return leaf value if stopping condition met
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree
myDat = loadDataSet('ex00.txt')
myMat = mat(myDat)
# print(createTree(myMat))
myDat1 = loadDataSet('ex0.txt')
myMat1 = mat(myDat1)
# print(createTree(myMat1))
# ops 参数调整树由本来的2个分支变成了很多个，不合适，so ops参数很重要
# print(createTree(myMat, ops=(0, 1)))
myDat2 = loadDataSet('ex2.txt')
myMat2 = mat(myDat2)
# print(createTree(myMat2, ops=(10000, 4)))
# Regression tree-pruning functions
def isTree(obj):
    return (type(obj).__name__ == 'dict')
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
# The method we’ll use will first split our data into a test set and a training set. First,
# you’ll build the tree with the setting that will give you the largest, most complex tree
# you can handle. You’ll next descend the tree until you reach a node with only leaves.
# You’ll test the leaves against data from a test set and measure if merging the leaves
# would give you less error on the test set. If merging the nodes will reduce the error on
# the test set, you’ll merge the nodes.
# Pseudo-code for prune() would look like this:
# Split the test data for the given tree:
# If the either split is a tree: call prune on that split
# Calculate the error associated with merging two leaf nodes
# Calculate the error without merging
# If merging results in lower error then merge the leaf nodes
# 用树的叶子节点的父节点去分割测试集，看分割前后的error的变化决定是否合并父节点，以减少overfitting!
def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            # 如果合并树的误差小于不合并的误差，那么返回合并的树，此时是一个叶子节点
            return treeMean
        else:
            # 否则返回原来的树，树中含左右两个节点
            return tree
    else:
        return tree
myTree = createTree(myMat2, ops=(0, 1))
myDatTest = loadDataSet('ex2test.txt')
myMat2Test = mat(myDatTest)
print(prune(myTree, myMat2Test))
# Leaf-generation function for model trees
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannnot do inverse, try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X*ws
    return sum(power(Y-yHat, 2))
myMat3 = mat(loadDataSet('exp2.txt'))
# print(createTree(myMat3, modelLeaf, modelErr, (1, 10)))
# Code to create a forecast with tree-based regression
def regTreeEval(model, inDat):
    return float(model)
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
# myTree1 = createTree(trainMat, ops=(1, 20))
# yHat = createForeCast(myTree1, testMat[:, 0])
# corrcoef 相关系数 ，返回yHat和testMat[:, 1]的相关系数
# print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
# --------------------------------------------------------
# myTree2 = createTree(trainMat, modelLeaf, modelErr, (1, 20))
# yHat = createForeCast(myTree2, testMat[:, 0], modelTreeEval)
# print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
# --------------------------------------------------------
# ws, X, Y = linearSolve(trainMat)
# print(ws)
# m, n = shape(trainMat)
# yHat = mat(ones((m, 1)))
# for i in range(shape(testMat)[0]):
#     yHat[i] = testMat[i, 0] * ws[1, 0]+ws[0, 0]
# print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])












