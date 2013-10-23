import preparation
import plotter
import utils
import knn

def datingClassTest():
    ratio=0.1
    k=3
    features,labels= preparation.file2matrix('datingTestSet2.txt')
    features=utils.normalize(features)
    #plotter.scatter(features[:,0],features[:,1],labels)
    numTestVecs=int(ratio*features.shape[0])
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = knn.classify(features[i,:],features[numTestVecs:features.shape[0],:],labels[numTestVecs:features.shape[0]],k)
        print "feature: %s | yhat: %d | y: %d" % (features[i,:],classifierResult,labels[i])
        if (classifierResult != labels[i]): errorCount+=1
    print "Error rate = %f" % (errorCount/numTestVecs)

datingClassTest()
