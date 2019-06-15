# The FP stands for “frequent pattern.”
# FP-tree class definition
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        # The nodeLink variable will be used to link similar items (the dashed lines in figure 12.1)
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
    def inc(self, numOccur):
        self.count += numOccur
    def disp(self, ind=1):
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)
# rootNode = treeNode('pyramid', 9, None)
# rootNode.children['eye'] = treeNode('eye', 13, None)
# rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
# print(rootNode.disp())
# FP-tree creation code
def createTree(dataSet, minSup=1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # removing items not meeting support
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    # if no items meet min support, exit
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        # the header table is slightly expanded so it can hold a count and pointer to the first item of each type
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        # sort transactions by global frequency
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # recursively call updateTree on remaining items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode
# Simple dataset and data wrapper
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict
simpDat = loadSimpDat()
initSet = createInitSet(simpDat)
myFPTree, myHeaderTab = createTree(initSet, 3)
# myFPTree.disp()
# to generate a conditional pattern base given a single item.
# A function to find all paths ending with a given item
# recursively ascend the tree
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats
# paths = findPrefixPath('r', myHeaderTab['r'][1])
# print(paths)
# The mineTree function recursively finds frequent itemsets.
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # sort: default sort is lowest to highest
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    # start from bottom of header table
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # construct cond.FP-tree from cond.pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        # if myCondTree != None:
        #     print('conditional tree for: ', newFreqSet)
        #     myCondTree.disp(1)
        if myHead != None:
            # mine cond.FP-tree
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

freqItems = []
mineTree(myFPTree, myHeaderTab, 3, set([]), freqItems)
# print(freqItems)
# encourage you to explore all functionality of the API. give up
# another example
parseDat = [line.split() for line in open('kosarak.dat').readlines()]
initSet1 = createInitSet(parseDat)
myFPTree1, myHeaderTab1 = createTree(initSet1, 100000)
myFreqList = []
mineTree(myFPTree1, myHeaderTab1, 100000, set([]), myFreqList)
print(len(myFreqList))
print(myFreqList)





