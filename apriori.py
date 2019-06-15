# Apriori algorithm helper functions
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
# # 创建只有一个元素的项集列表；即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
def createC1(dataSet):
    # C1 is a candidate itemset of size one
    # param :dataSet
    # return:frozenset 返回一个 frozenset 格式的 list
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            # #遍历所有元素，添加进入“单个物品的项集列表”
            if not [item] in C1:
                # #注意！！！item一定要加中括号，代表列表； 不然C1的元素是int，int是不可迭代的；
                # 执行list(map(frozenset, C1))，报错如下：TypeError: 'int' object is not iterable
                C1.append([item])
    C1.sort()
    # 对每一个元素 frozenset
    return list(map(frozenset, C1))
# 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度（minSupport）的数据（即频繁项集）
# Ck, a list of candidate sets, and minSupport, which is the minimum support you’re interested in.
def scanD(D, Ck, minSupport):
    # 临时存放选数据集 Ck 的频率.
    ssCnt = {}
    # # 遍历数据集中每一条交易记录
    for tid in D:
        # 遍历每一项候选项集
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []  # 支持度大于 minSupport 的集合
    supportData = {}   # 候选项集支持度数据
    for key in ssCnt:
        # 支持度 = 候选项（key）出现的次数 / 所有数据集的数量
        support = ssCnt[key]/numItems
        if support >= minSupport:
            # 在 retList 的首位插入元素，只存储支持度满足频繁项集的值
            retList.insert(0, key)
        # 存储所有的候选项（key）和对应的支持度（support）
        supportData[key] = support
    return retList, supportData
dataSet = loadDataSet()
# The Apriori algorithm
# create Ck
# The function aprioriGen() will take a list of frequent itemsets, Lk, and the size of the itemsets, k, to produce Ck.
# For example, it will take the itemsets {0}, {1}, {2} and so on and produce {0,1} {0,2}, and {1,2}.
# 输入频繁项集列表 Lk， k：返回的元素个数， 然后输出所有可能的候选项集 Ck
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # [:k-2] 表示0-k-3的元素，亦即Lk中除最后一个数之外的所有链表元素。
            # 若最后一个数不等，合并两个集合，合并后的集合元素数比Lk中的元素数+1
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            # join sets if first k-2 items are equal
            # 第一次 L1,L2 为空，元素直接进行合并，返回元素两两合并的数据集
            if L1 == L2:
                # The sets are combined using the set union, which is the | symbol in Python.
                retList.append(Lk[i] | Lk[j])
    return retList
def apriori(dataSet, minSupport=0.5):
    #  首先构建集合 C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。
    # 那么满足最小支持度要求的项集构成集合 L1。然后 L1 中的元素相互组合成 C2，C2 再进一步过滤变成 L2，
    # 然后以此类推，知道 CN 的长度为 0 时结束，即可找出所有频繁项集的支持度。）
    #     :param dataSet: 原始数据集
    #     :param minSupport: 支持度的阈值
    #     :return:
    #         L 频繁项集的全集
    #         supportData 所有元素和支持度的全集
    # 只有一个元素的候选项集列表；即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
    C1 = createC1(dataSet)
    # 对每一行进行 set 转换，然后存放到集合中,并转换成列表
    D = list(map(set, dataSet))
    # 计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
    L1, supportData = scanD(D, C1, minSupport)
    # 给 L 加了一层 list, 本来的 L 现在变成了  L[0]
    L = [L1]
    k = 2
    # 判断 L 的第 k-2 项的数据长度是否 > 0。最后一个元素为[] = 0，时表示无法继续合并，满足条件的元素集已经全部合并
    # 第一次执行时 L 为 [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]]。
    # L[k-2]=L[0]=[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]，最后面 k += 1
    # 用L[k-2]中的元素合并新的support集
    while (len(L[k-2]) > 0):
        # call aprioriGen() to create candidate itemsets: Ck.
        # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}.
        # 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
        Ck = aprioriGen(L[k-2], k)
        # scan data set to get Lk from Ck
        # 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据，即为Lk
        Lk, supK = scanD(D, Ck, minSupport)
        # 保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
        supportData.update(supK)
        # Lk 表示满足频繁子项的集合，L 元素在增加，例如:
        # l=[[set(1), set(2), set(3)]]
        # l=[[set(1), set(2), set(3)], [set(1, 2), set(2, 3)]]
        # L的第一个元素是包含“只含有一个元素”频繁项集列表；L的第二个元素是包含有“两个元素的”频繁项集列表.....
        L.append(Lk)
        k += 1
    return L, supportData
L, suppData = apriori(dataSet)
# print(L)
# Association rule-generation functions
# L: a list of frequent itemsets,
# supportData: a dictionary of support data for those itemsets,
# minConf: and a minimum confidence threshold
# bigRuleList: generate a list of rules with confidence values that we can sort through later
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList
# evaluate those rules,
def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            br1.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH
# generate a set of candidate rules
def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    # try further merging
    if (len(freqSet) > (m+1)):
        # create Hm+1 new candidates
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)

rules = generateRules(L, suppData, minConf=0.5)
# print(rules)
mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
Lmr, suppDatamr = apriori(mushDatSet, minSupport=0.3)
for item in Lmr[1]:
    if item.intersection('2'):
        print(item)
















