from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids
datMat = mat(loadDataSet('testSet10.txt'))
# print(distEclud(datMat[0], datMat[1]))
# The k-means clustering algorithm
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    # clusterAssment, has two columns;
    # one column  is  for  the  index  of  the  cluster
    # and  the  second  column  is  to  store  the  error.
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 现在在的cluster是否是 离最近的Index(minIndex相等，如果不是，就改变其所在的Cluster，继续循环
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        # This is done by first doing  some  array  filtering  to  get  all  the  points  in  a  given  cluster.
        # Next,  you  take  the mean values of all these points.
        # The option axis=0 in the mean calculation does the mean  calculation  down  the  columns.
        # Finally,  the  centroids  and  cluster  assignments are returned.
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment
# myCentroids, clustAssing = kMeans(datMat, 4)
# The bisecting k-means clustering algorithm
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    # one  centroid  is  calculated  for  the entire dataset,
    # and a list is created to hold all the centroids.
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    # clusterAssment, has two columns;
    # one column  is  for  the  index  of  the  cluster
    # and  the  second  column  is  to  store  the  error.!!
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
    # splits  clusters  until  you  have  the  desired number of clusters
    while (len(centList) < k):
        lowestSSE = inf
        # iterate over all the clusters and find the best cluster to split
        for i in range(len(centList)):
            # create a dataset of only the points from that cluster
            #  called ptsInCurrCluster, fed into kMeans()
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            #  The k-means algorithm gives you two new centroids as well as the squared
            # error for each of those centroids.
            #  These errors are added up(sseSplit) along with the error(sseNotSplit) for the rest of the dataset.
            # If this split produces the lowest  SSE , then it’s saved
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit:", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # When  you  applied kMeans() with  two  clusters,
        # you  had  two  clusters  returned labeled  0  and  1.
        # You need to change these cluster numbers to the cluster number you’re splitting
        # and the next cluster to be added.
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print("the bestCentToSplit is:", bestCentToSplit)
        print("the len of bestClustAss is:", len(bestClustAss))
        # Finally, these new cluster assignments are updated and the new centroid is appended to centList
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        # update the assignments ,bestCentToSplit为被split的cluster，属于这个cluster的都要更新！
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssment
datMat3 = mat(loadDataSet('testSet10_2.txt'))
centList, myNewAssments = biKmeans(datMat3, 3)
print(centList)







