from math import log

def entropy(dataSet):
    numEntries=len(dataSet)
    labels={}
    for data in dataSet:
        label=data[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label]+=1
    ent=0.0
    for key in labels:
        prob = float(labels[key])/numEntries
        ent-=prob*log(prob,2)
    return ent
