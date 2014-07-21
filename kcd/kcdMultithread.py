#!/usr/bin/python

import math
import logging
import pandas as pd
import numpy as np
from sklearn import svm
import sklearn.metrics

# from multiprocessing.dummy import Pool
from multiprocessing import Pool

def convertDataToCov(data, windowSize=75):

    covariances = []

    for i in range(windowSize, data.shape[0]):

        windowData = data[i-windowSize:i]
        zeroedData = windowData - windowData.mean()

        covMatrix = np.dot(zeroedData.T, zeroedData) / float(windowSize - 1)

        covariances.append(covMatrix.ravel())

    return pd.DataFrame(covariances)

def singlePointDetector(packedArgument):

    (df, d, targetT, gamma, nu, useCov) = packedArgument

    start = targetT - d
    end = targetT + d
    
    leftData = df[start:targetT]
    rightData = df[targetT:end]

    # This seems to work the best for cov
    if ( useCov == True ):
        leftData = convertDataToCov(leftData, windowSize=d/2)
        rightData = convertDataToCov(rightData, windowSize=d/2)

    leftSvm = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    rightSvm = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)

    leftSvm.fit(leftData.as_matrix())
    rightSvm.fit(rightData.as_matrix())

    alphaLeft = np.zeros((leftData.shape[0],1))
    for i in range(leftSvm.support_.shape[0]):
        alphaLeft[leftSvm.support_[i]][0] = leftSvm.dual_coef_[0][i]

    alphaRight = np.zeros((rightData.shape[0],1))
    for i in range(rightSvm.support_.shape[0]):
        alphaRight[rightSvm.support_[i]][0] = rightSvm.dual_coef_[0][i]

    m = leftData.shape[0]

    k11 = sklearn.metrics.pairwise.rbf_kernel(leftData.as_matrix(), leftData.as_matrix(), gamma=gamma)
    k22 = sklearn.metrics.pairwise.rbf_kernel(rightData.as_matrix(), rightData.as_matrix(), gamma=gamma)
    k12 = sklearn.metrics.pairwise.rbf_kernel(leftData.as_matrix(), rightData.as_matrix(), gamma=gamma)

    top = np.dot(np.dot(alphaLeft.T, k12), alphaRight)
    botLeft = np.sqrt(np.dot(np.dot(alphaLeft.T, k11), alphaLeft))
    botRight = np.sqrt(np.dot(np.dot(alphaRight.T, k22), alphaRight))

    ct1ct2 = np.arccos(top / (botLeft * botRight))
    ct1pt1 = np.arccos(leftSvm.intercept_/botLeft)
    ct2pt2 = np.arccos(rightSvm.intercept_/botRight)

    dH = (ct1ct2 / (ct1pt1 + ct2pt2))[0][0]

    return (targetT, dH)

def kernelChangeDetection(df, d=50, eta=0.4, gamma=0.25, nu=0.0625, useCov=False):

    kcdStat = []
    changePoints = []

    # Pool for SVM algorithms
    pool = Pool()

    taskList = [(df, d, target, gamma, nu, useCov) for target in range(d,df.shape[0] - d)]

    print "Task List Length:", len(taskList)

    mapResults = pool.map(singlePointDetector, taskList)

    for targetT, dH in mapResults:
                
        kcdStat.append(dH)

        if ( dH > eta ):
            changePoints.append(targetT)

    return (changePoints, kcdStat)
    
if __name__ == '__main__':

    import sys
    import json

    if (len(sys.argv) < 2):
        print "Usage: %s <input_file>" % sys.argv[0]
        exit(1)

    dataFile = sys.argv[1]

    useCov = True
    df = None
    k = None
    if (dataFile.endswith('.mat')):
        # Matlab file...
        from scipy.io import loadmat
        inputMatlabData = loadmat(dataFile)
        matlabData = inputMatlabData['simData']

        realizations = 5000
        k = matlabData.shape[1] / realizations
        i = 1

        # df = pd.DataFrame(matlabData[:,k*i:k*i+k])
        df = pd.DataFrame(matlabData)

        # Use covariance rather than data
        useCov = True
    elif (dataFile.endswith('.csv')):
        # CSV file
        df = pd.read_csv(dataFile, header=0)

        if ( type(df[df.columns[0]][0]) == str ):
            print "Reindexing using column:", df.columns[0]
            df['index'] = pd.DatetimeIndex(df[df.columns[0]])
            df = df.set_index('index')
            df = df[df.columns[1:]]
            df = df.sort_index()

        k = df.shape[1]
    else:
        print "Unknown file type:", dataFile
        exit(1)

    print df

    # (changePoints, kcdStat) = kernelChangeDetection(df, d=50, eta=0.5, nu=0.125, gamma=0.25)
    (changePoints, kcdStat) = kernelChangeDetection(df, d=30, eta=4.1, nu=0.125, gamma=0.005, 
        useCov=useCov)

    print "Found Change Points:", changePoints
    for t in df.index[changePoints]:
        print t

    outputFilename = dataFile + '.json'
    outputFile = open(outputFilename, 'w')
    outputFile.write(json.dumps({"data":kcdStat, "changepoints":changePoints}))
    outputFile.close()

    import pylab as pl
    pl.figure(figsize=(8, 6), dpi=80)
    pl.plot(kcdStat)
    pl.show()