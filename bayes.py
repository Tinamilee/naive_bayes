from numpy import *
import re
import operator
import feedparser

#作为简单数据集练习，返回两个值，一个数据集和其对应的标签类型，1表示spam，0表示not spam
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVes = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVes

# 创建数据集中出现过的词典表，set方法可去除数组中重复出现的元素，
# '|' 表示‘或’方法，将已有元素和新添加的元素，不重复的合并到vocabSet中去
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 返回对应字典数组中的字典向量，index对应，1表示该位置中字是否在总的字典集vocabList出现过
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        # else:
            # print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 返回对应字典数组中的字典向量，index对应，值表示该位置中字在总的字典集vocabList出现过的次数，每出现一次就+1
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# listOPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
# print(myVocabList)
# x = setOfWords2Vec(myVocabList, listOPosts[0])
# y = setOfWords2Vec(myVocabList, listOPosts[3])
# print(x)
# print(y)

# 训练朴素贝叶斯算法，函数接收训练集和其对应的标签集
# the reason of p0Num = ones(numWords), p0Denom = 2.0
#  This will look somethinglikep(w0|1)p(w1|1)p(w2|1).  If  any  of  these  numbers  are  0,  then  when  we  multiply
# them together we get 0. To lessen the impact of this, we’ll initialize all of our occur-
# rence counts to 1, and we’ll initialize the denominators to 2.

# 0*any=0，值太小的话python会round-off  -->  0所以也可以用log方法解决问题
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)      # 训练集tuple的个数， 训练子集是0，1的（1，numWords），表示词是否在这个子集出现
    numWords = len(trainMatrix[0])       # 总共单词数
    pAbusive = sum(trainCategory)/float(numTrainDocs)       # pAbusive表示垃圾邮件在训练集中出现的概率
    p0Num = ones(numWords)                  # 初始化为（1，numWords）的shape的 值为1的一维数组
    p1Num = ones(numWords)                  # 不初始化为0是因为如上
    p0Denom = 2.0                           # 初始化为2，也是因为值太小
    p1Denom = 2.0
    for i in range(numTrainDocs):           # 循环遍历所有训练集，numTrainDocs为训练子集个数
        if trainCategory[i] == 1:           # 如果子集的标签为1（spam）
            p1Num += trainMatrix[i]         # 累加到p1Num矩阵中，对应位置表示该词在spam中出现的次数
            p1Denom += sum(trainMatrix[i])  # 累加，将子集中出现的词的次数和存到p1Denom中，表示spam中所有出现过词的次数和
        else:                               # 如果子集的标签为0（not spam）
            p0Num += trainMatrix[i]         # 累加到p0Num矩阵中，对应位置表示该词在not spam中出现的次数
            p0Denom += sum(trainMatrix[i])  # 累加，将子集中出现的词的次数和存到p0Denom中，表示not spam中所有出现过词的次数和
    p1Vect = p1Num/p1Denom      # 将spam中词出现次数的矩阵除以spam中所有出现过词的次数得到的矩阵就是对应词出现的话，该邮件为spam的概率
    p0Vect = p0Num/p0Denom      # 将0标签中词出现次数的矩阵除以not spam中所有出现过词的次数得到的矩阵就是对应词出现的话，该邮件为not spam的概率
    return p0Vect, p1Vect, pAbusive        # 返回概率数组表示对应位置词出现该邮件为notspam和spam的概率，以及邮件在总训练集中为spam的概率

# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
# p0V, p1V, pAb = trainNB0(trainMat, listClasses)
# print(pAb)
# print(p0V)
# print(p1V)
# 朴素贝叶斯算法分类，接收被分类数据和两个标签概率数组以及spam概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)         # 计算该数据为not or spam邮件的概率，vec2Classify表示对应位置的词在被测试数据中出现的次数
    p0 = sum(vec2Classify*p0Vec) + log(1.0 - pClass1)   # * 对应概率数组表示该词出现在对应标签类的概率 加上对应log（标签概率）表示所求概率
    if p1 > p0:                                         # 判断被测试数据为spam 和 not spam中哪个概率大
        return 1                                        # 返回较大概率的标签
    else:
        return 0
# 测试朴素贝叶斯算法
def testingNB():
    listOPosts, listClasses = loadDataSet()             # 加载数据
    myVocabList = createVocabList(listOPosts)           # 创建数据集中的词典表
    trainMat=[]                                         # 定义训练集
    for postinDoc in listOPosts:                        # 循环遍历数据集
        # trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))   # 将处理好的1，0数组存到trainMat中，表示该位置的词出现过的次数
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))   # 训练朴素贝叶斯算法将结果存起来
    testEntry = ['love', 'my', 'dalmation', 'stupid']               # 测试数据testEntry
    # thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))       # 将测试数据转成可操作的0， 1数组，表示该位置的词出现过的次数
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))  # 输出该数据被朴素贝叶斯算法分类的结果
    testEntry = ['stupid', 'garbage', 'dog']                        # 新的测试数据，同上一个操作一样
    # thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

# testingNB()
# regEx = re.compile('\\W*')
# emailText = open('email/ham/6.txt').read()
# listOfTokens = regEx.split(emailText)
# 文本操作
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)                  # 定义正则化表达式，\W* 表示匹配任意非单词字符，作为分隔符
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 返回分割后长度大于2的元素，去除空格，ny，sy等无实际意义的字符串
# spam测试 输出错误率
def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1, 26):      # 因为ham和spam文件夹中每个有25个文件，标号 1-25
        fp = open('email/spam/%d.txt' % i, 'rb')    # 打开第i个文件，用rb二进制的方式，因为有？￥$等字符，会出问题
        s = fp.read().decode('ISO-8859-15')         # 打开之后，重新以ISO-8859-15（txt文本编码方式）编码还原数据
        wordList = textParse(s)                     # 将打开文本后得到的字符串转化为可处理的字符数组格式
        docList.append(wordList)                    # 将所有处理好的数据加入docList，无论spam与否
        fullText.extend(wordList)                   # 将处理好的数据加入fullText中，（后来到计算词在总集中出现的次数才用到）
        classList.append(1)                         # 将spam文件中的数据class置为1，
        fp = open('email/ham/%d.txt' % i, 'rb')
        s = fp.read().decode('ISO-8859-15')
        wordList = textParse(s)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)   #返回总数据集中的单词表

    # top30Words = calcMostFreq(vocabList, fullText)
    # for pairW in top30Words:
    #     if pairW[0] in vocabList:
    #         vocabList.remove(pairW[0])   删除后下标会变，不适用于spam中关于index的各种操作

    trainingSet = list(range(50))       #总共有50个数据，处理index分类train和test集
    testSet=[]
    for i in range(10):                 # 随机取出十个不一样的index加入test，并从train中剔除
        randIndex = int(random.uniform(0, len(trainingSet)))   # uniform方法可实现取不同index
        testSet.append(trainingSet[randIndex])                 # 将该index下的数据给到test中
        del(trainingSet[randIndex])                            # 并将该index下的数据从train中剔除
    trainMat = []                                              # 定义训练数据和训练标签
    trainClasses = []
    for docIndex in trainingSet:                               # 循环遍历训练集
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))         # 将数据转化为表示该位置的词出现的次数的数组
        trainClasses.append(classList[docIndex])                    # 将对应数据的class加入到trainClass中
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))   # 训练朴素贝叶斯算法，并存其结果
    errorCount = 0                                              # 计数错误的次数
    for docIndex in testSet:                                    # 循环遍历testSet
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])  # 将被测试的数据转化为词的出现次数的数据
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 判断朴素贝叶斯算法是否分类正确
            errorCount += 1                                                         # 分类错误的话errorCount+1
            # print(docList[docIndex])
            # print(classList[docIndex])
    print("the error rate is: ", float(errorCount)/len(testSet))        # 将总分类错误次数/总测试数为错误率
    # return vocabList, p0V, p1V

spamTest()
def calcMostFreq(vocabList, fullText):     # 找出出现频率较高的词
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)         #freqDict存储token在总数据集中出现的次数
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)  # 按照出现次数从高到低排序
    return sortedFreq[:30]   # 返回前30个出现词数较高的词

# 和spamTest类似，但将较高频率出现的词从vocabList中剔除，错误率会低点
def localWords(feed1, feed0):
    docList=[]
    classList=[]
    fullText=[]
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'[i]['summary']])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen)
    testSet=[]
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docIndex[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is: ", float(errorCount)/len(testSet))
    return vocabList, p0V, p1V
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')  # 用feedparser模块从网上解析rss文件，
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')      # 本例子中两个网址取不出数据gg
# vocabList, pSF, pNY = localWords(ny, sf)

# 返回频率较高的词
def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY=[]
    topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > 0.015:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > 0.015:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
# getTopWords(ny, sf)





