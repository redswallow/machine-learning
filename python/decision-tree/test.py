import id3
import entropy

def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

myDat,labels=createDataSet()
#print entropy.entropy(myDat)
#print id3.chooseBestFeatureToSplit(myDat)
tree = id3.createTree(myDat,labels)
print tree
