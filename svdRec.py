from numpy import *
from numpy import linalg as la
def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]
# Data = loadExData()
# U, Sigma, VT = linalg.svd(Data)
# print(Sigma)
# Sigma3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
# print(U[:, :3] * Sigma3 * VT[:3, :])
# Similarity measures
def ecludSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))
def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]
def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5*(num/denom)
# myMat = mat(Data)
# print(ecludSim(myMat[:, 0], myMat[:, 4]))
# print(cosSim(myMat[:, 0], myMat[:, 4]))
# print(pearsSim(myMat[:, 0], myMat[:, 4]))
# Item-based recommendation engine
# calculates the estimated rating a user would give an item for a given similarity measure.
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        # Find items rated by both users
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
            # print('the %d and %d similarity is: %f' % (item, j, similarity))
            simTotal += similarity
            ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # find unrated items
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    # return top N unrated items
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]
def loadExData1():
    return [[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]]
# myMat1 = mat(loadExData1())
# print(recommend(myMat1, 2))
# print(recommend(myMat1, 2, simMeas=ecludSim))
# print(recommend(myMat1, 2, simMeas=pearsSim))
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
def loadExData3():
    # 利用SVD提高推荐效果，菜肴矩阵
    """
    行：代表人
    列：代表菜肴名词
    值：代表人对菜肴的评分，0表示未评分
    """
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]
myMat = mat(loadExData2())
U, Sigma, VT = la.svd(myMat)
# print(Sigma)
# Sig2 = Sigma**2
# print(sum(Sig2))
# sum(Sig2[:3]) = sum(Sig2)*0.9,so reduce to a 3-dimensional matrix
# Rating estimation by using the SVD
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    # it does an SVD on the dataset on the third line.
    U, Sigma, VT = la.svd(dataMat)
    # create diagonal matrix
    # After the SVD is done, you use only the singular values that give you 90% of the energy.
    Sig4 = mat(eye(4)*Sigma[:4])
    # create transformed items
    # use the U matrix to transform our items into the lower-dimensional space.
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal
# recommend(myMat, 1, estMethod=svdEst)
# recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim)
# Image-compression functions
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1)
            else:
                print(0)
        print('')
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("***original matrix***")
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("***reconstructed matrix using %d singular values***" % numSV)
    printMat(reconMat, thresh)
imgCompress(2)















