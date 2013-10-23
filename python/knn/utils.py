from numpy import *

def normalize(dataSet):
    ranges = dataSet.max(0) - dataSet.min(0)
    normDataSet = zeros(shape(dataSet))
    normDataSet = dataSet - tile(dataSet.min(0), (dataSet.shape[0],1))
    normDataSet = normDataSet/tile(ranges, (dataSet.shape[0],1))
    return normDataSet
