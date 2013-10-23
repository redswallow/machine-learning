from numpy import *
import operator

def classify(inX, dataSet, labels, k):
    """ knnClassifier
    """
    dataSetSize=dataSet.shape[0]
    sqDiff=(tile(inX,(dataSetSize,1))-dataSet)**2
    distances=sqDiff.sum(axis=1)**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
