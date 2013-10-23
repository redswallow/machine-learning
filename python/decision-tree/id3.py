from entropy import entropy
import operator

def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for feature in dataSet:
        if feature[axis]==value:
            reducedFeature = feature[:axis]
            reducedFeature.extend(feature[axis+1:])
            retDataSet.append(reducedFeature)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = entropy(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in xrange(numFeatures):
        InfoGain=baseEntropy
        #x=getColumn(dataSet,i)
        x= [sample[i] for sample in dataSet]
        for k in set(x):
            subDataSet = splitDataSet(dataSet, i, k)
            xProb=float(len(subDataSet)/len(dataSet))
            xEntropy=entropy(subDataSet)
            InfoGain-=xProb*xEntropy
            print i,k,xProb,xEntropy
        if InfoGain>bestInfoGain:
            bestInfoGain=InfoGain;bestFeature=i
    return bestInfoGain,bestFeature

def majorityVote(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [sample[-1] for sample in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityVote(classList)
    bestInfoGain,bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    
    decisionTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    feature = [sample[bestFeature] for sample in dataSet]
    for value in set(feature):
        subLabels = labels[:]
        decisionTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value),subLabels)
    
    return decisionTree
