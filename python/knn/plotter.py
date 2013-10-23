import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import operator

def scatter(x,y,labels=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if labels:
        ax.scatter(x,y,15*array(labels),15*array(labels))
    else:
        ax.scatter(x, y)
    plt.show()
