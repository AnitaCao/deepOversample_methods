# For running in python 2.x
#from __future__ import print_function, unicode_literals
#from __future__ import absolute_import, division

import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

def relabel(labelTr, labelTs):
    unqLab, pInClass=np.unique(labelTr, return_counts=True) #unique labels and counts
    sortedUnqLab=np.argsort(pInClass, kind='mergesort') #sort unique labels based on counts
    c=sortedUnqLab.shape[0] #number of classes
    labelsNewTr=np.zeros((labelTr.shape[0],))-1 #initialize new labels for training data
    labelsNewTs=np.zeros((labelTs.shape[0],))-1 #initialize new labels for test data
    pInClass=np.sort(pInClass) #sort counts
    classMap=list()
    for i in range(c):
        labelsNewTr[labelTr==unqLab[sortedUnqLab[i]]]=i #assign new labels for training data
        labelsNewTs[labelTs==unqLab[sortedUnqLab[i]]]=i #assign new labels for test data
        classMap.append(np.where(labelsNewTr==i)[0]) #store indices of each class
    return labelsNewTr, labelsNewTs, c, pInClass, classMap

def irFind(pInClass, c, irIgnore=1):
    ir=pInClass[-1]/pInClass #imbalance ratio
    imbalancedCls=np.arange(c)[ir>irIgnore] #imbalanced classes
    toBalance=np.subtract(pInClass[-1], pInClass[imbalancedCls]) #number of samples to balance
    imbClsNum=toBalance.shape[0] #number of imbalanced classes
    if imbClsNum==0: sys.exit('No imbalanced classes found, exiting ...')
    return imbalancedCls, toBalance, imbClsNum, ir

def fileRead(fileName):
    dataTotal=np.loadtxt(fileName, delimiter=',')
    data=dataTotal[:, :-1]
    labels=dataTotal[:, -1]
    return data, labels

def indices(pLabel, tLabel):
    confMat=confusion_matrix(tLabel, pLabel)
    nc=np.sum(confMat, axis=1)
    tp=np.diagonal(confMat)
    tpr=tp/nc
    acsa=np.mean(tpr)
    gm=np.prod(tpr)**(1/confMat.shape[0])
    acc=np.sum(tp)/np.sum(nc)
    return acsa, gm, tpr, confMat, acc

def randomLabelGen(toBalance, batchSize, c):
    cumProb=np.cumsum(toBalance/np.sum(toBalance))
    bins=np.insert(cumProb, 0, 0)
    randomValue=np.random.rand(batchSize,)
    randLabel=np.digitize(randomValue, bins)-1
    randLabel_cat=to_categorical(randLabel)
    labelPadding=np.zeros((batchSize, c-randLabel_cat.shape[1]))
    randLabel_cat=np.hstack((randLabel_cat, labelPadding))
    return randLabel_cat

def batchDivision(n, batchSize):
    numBatches, residual=int(np.ceil(n/batchSize)), int(n%batchSize)
    if residual==0:
        residual=batchSize
    batchDiv=np.zeros((numBatches+1,1), dtype='int64')
    batchSizeStore=np.ones((numBatches, 1), dtype='int64')
    batchSizeStore[0:-1, 0]=batchSize
    batchSizeStore[-1, 0]=residual
    for i in range(numBatches):
        batchDiv[i]=i*batchSize
    batchDiv[numBatches]=batchDiv[numBatches-1]+residual
    return batchDiv, numBatches, batchSizeStore

#convert one-hot encoded labels to original labels
def rearrange(labelsCat, numImbCls):
    labels=np.argmax(labelsCat, axis=1)
    arrangeMap=list()
    for i in range(numImbCls):
        arrangeMap.append(np.where(labels==i)[0])
    return arrangeMap