#!/usr/bin/python

import math
import logging
import pandas as pd
import numpy as np
from sklearn import svm
import sklearn.metrics

from multiprocessing.pool import ThreadPool

def convertDataToCov(data, windowSize=75):

    covariances = []

    for i in range(windowSize, data.shape[0]):

        windowData = data[i-windowSize:i]
        zeroedData = windowData - windowData.mean()

        covMatrix = np.dot(zeroedData.T, zeroedData) / float(windowSize - 1)

        covariances.append(covMatrix.ravel())

    return pd.DataFrame(covariances)

def kernelChangeDetection(df, d=50, eta=0.4, gamma=0.25, nu=0.0625):

    kcdStat = []
    changePoints = []

    for targetT in range(d,df.shape[0] - d):
        
        start = targetT - d
        end = targetT + d
        
        leftData = df[start:targetT]
        rightData = df[targetT:end]

        # leftData = convertDataToCov(leftData, windowSize=d/2)
        # rightData = convertDataToCov(rightData, windowSize=d/2)
        
        # Try and model the time series as VAR1
        # TODO: Remove if no benefit
    #     leftModel = tsa.vector_ar.var_model.VAR(leftData.as_matrix())
    #     leftFit = leftModel.fit()
    #     leftData = pd.DataFrame(leftFit.resid)
    #     rightModel = tsa.vector_ar.var_model.VAR(rightData.as_matrix())
    #     rightFit = rightModel.fit()
    #     rightData = pd.DataFrame(rightFit.resid)

        leftSvm = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
        rightSvm = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)

        leftSvm.fit(leftData.as_matrix())
        rightSvm.fit(rightData.as_matrix())
        
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     b1 = ax.scatter(leftData.icol(0), leftData.icol(1), leftData.icol(2), c='red')
    #     b2 = ax.scatter(rightData.icol(0), rightData.icol(1), rightData.icol(2), c='white')
    #     plt.show()

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
        ct1pt1 = np.arccos(np.abs(leftSvm.intercept_)/(botLeft))
        ct2pt2 = np.arccos(np.abs(rightSvm.intercept_)/(botRight))

        dH = (ct1ct2 / (ct1pt1 + ct2pt2))[0][0]
        
        kcdStat.append(dH)

        if ( dH > eta ):
            changePoints.append(targetT)

    return (changePoints, kcdStat)
    
if __name__ == '__main__':

    import sys
    import json

    if (len(sys.argv) < 2):
        print "Usage: %s <csv_file>" % sys.argv[0]
        exit(1)

    dataFile = sys.argv[1]
    df = pd.read_csv(dataFile, header=0)

    if ( type(df[df.columns[0]][0]) == str ):
        print "Reindexing using column:", df.columns[0]
        df['index'] = pd.DatetimeIndex(df[df.columns[0]])
        df = df.set_index('index')
        df = df[df.columns[1:]]
        df = df.sort_index()

    print df

    indexlessDf = df[df.columns[1:]]

    # import cProfile
    # profiledFunc = lambda: kernelChangeDetection(indexlessDf, d=100, eta=0.35, nu=0.5)
    # cProfile.run('profiledFunc()', 'kcd.profile')
    # exit(1)

    stats = []

    # for param in [0.005, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]:
    # for param in [25, 30, 50, 75, 100, 200]:
    for param in [None]:

        print param
        # (changePoints, kcdStat) = kernelChangeDetection(indexlessDf, d=50, eta=0.5, nu=0.125, gamma=0.25)
        (changePoints, kcdStat) = kernelChangeDetection(indexlessDf, d=200, eta=0.5, nu=0.125, gamma=0.25)

        stats.append((kcdStat, param))

        print "Found Change Points:", changePoints
        for t in df.index[changePoints]:
            print t

        outputFilename = dataFile + '.json'
        outputFile = open(outputFilename, 'w')
        outputFile.write(json.dumps({"data":kcdStat, "changepoints":changePoints}))
        outputFile.close()

    import pylab as pl
    pl.figure(figsize=(8, 6), dpi=80)
    for v in stats:
        pl.plot(v[0], label=str(v[1]))
    pl.legend()
    pl.show()