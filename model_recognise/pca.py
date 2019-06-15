# The PCA algorithm
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [list(map(float, line)) for line in stringArr]
    return mat(dataArr)
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    # remove mean
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    # sort stop N smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    # Transform data into new dimensions
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat
dataMat = loadDataSet('testSet_13.txt')
lowDMat, reconMat = pca(dataMat, 1)
shape(lowDMat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
ax.scatter(reconMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
plt.show()
# Function to replace missing values with mean
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # find mean of non-NaN values
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        # set NaN values to mean
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat
dataMat1 = replaceNanWithMean()
meanVals = mean(dataMat1, axis=0)
meanRemoved = dataMat1 - meanVals
covMat = cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))
print(eigVals)













