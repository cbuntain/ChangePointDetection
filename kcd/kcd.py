#!/usr/bin/python

import math
import logging
import pandas as pd
import numpy as np
from sklearn import svm

def rbfKernelFunc(x, y, gamma):
    z = np.zeros((x.shape[0], y.shape[0]))

    for i in range(x.shape[0]):
        xi = x[i]
        for j in range(y.shape[0]):
            yj = y[j]

            diff = (xi - yj)
            z[i][j] = np.exp(-1.0 * gamma * np.abs(np.dot(diff, diff.T)))

    return z

def kernelChangeDetection(df, d=50, eta=0.4, gamma=0.25, nu=0.0625):

    kcdStat = []
    changePoints = []

    for targetT in range(d,df.shape[0] - d):
        
        if ( targetT % 100 == 0 ):
            print targetT, "/", df.shape[0] - d
        
        start = targetT - d
        end = targetT + d
        
        leftData = df[start:targetT]
        rightData = df[targetT:end]
        
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

        k11 = rbfKernelFunc(leftData.as_matrix(), leftData.as_matrix(), gamma)
        k22 = rbfKernelFunc(rightData.as_matrix(), rightData.as_matrix(), gamma)
        k12 = rbfKernelFunc(leftData.as_matrix(), rightData.as_matrix(), gamma)

        top = np.dot(np.dot(alphaLeft.T, k12), alphaRight)
        botLeft = np.sqrt(np.dot(np.dot(alphaLeft.T, k11), alphaLeft))
        botRight = np.sqrt(np.dot(np.dot(alphaRight.T, k22), alphaRight))

        ct1ct2 = np.arccos(top / (botLeft * botRight))
        ct1pt1 = np.arccos(leftSvm.intercept_/botLeft)
        ct2pt2 = np.arccos(rightSvm.intercept_/botRight)

        dH = (ct1ct2 / (ct1pt1 + ct2pt2))[0][0]
        
        kcdStat.append(dH)

        if ( dH > eta ):
            changePoints.append(targetT)

    return (changePoints, kcdStat)
    
if __name__ == '__main__':

    import sys

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

    (changePoints, kcdStat) = kernelChangeDetection(indexlessDf, d=100, eta=0.35, nu=0.5)

    print "Found Change Points:", changePoints
    for t in df.index[changePoints]:
        print t

    import pylab as pl
    pl.figure(figsize=(8, 6), dpi=80)
    pl.plot(kcdStat)
    pl.show()