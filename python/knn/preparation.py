from numpy import *
import operator

def createDataSet():
    """ creates the dataset and labels
    """
    features = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return features, labels

def file2matrix(filename):
    f=open(filename)
    features = zeros((len(f.readlines()),3))
    labels = []
    index = 0
    f=open(filename)
    for line in f.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        features[index,:] = listFromLine[0:3]
        labels.append(int(listFromLine[-1]))
        index += 1
    return features, labels
