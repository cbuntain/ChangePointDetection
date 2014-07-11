#!/usr/bin/python

import math
import pandas
import logging
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa as tsa
import pylab as pl

import critValueSim

logger = logging.getLogger('driver')

def calculateS(residuals, offset, length):
    
    sMatrix = np.zeros((residuals.shape[1], residuals.shape[1]))
    
    for i in range(offset, offset+length):
        error = np.reshape(residuals[i], (residuals.shape[1], 1))
        errorCov = np.dot(error, error.T)

        sMatrix += errorCov

    sMatrix = (1./float(length)) * sMatrix
    
    return sMatrix

def recursiveChangePointDetector(residuals, p, q, k, index, results, alpha=0.001):

    d = k * (p + q + 1) + (k*(k+1)/2) + 1
    
    n = residuals.shape[0]

    # Pre-calculate covariances
    covariances = []
    for i in range(n):
    	error = np.reshape(residuals[i], (k, 1))
    	errorCov = np.dot(error, error.T)
    	covariances.append(errorCov)
    
    likelihoodRatioStats = np.zeros(n)
    sF = calculateS(residuals, 0, n)
    sFullDet = np.abs(np.linalg.det(sF))

    lastS1 = np.zeros((k,k))
    lastS2 = float(n) * sF

    sMatrixList = []

    for i in range(n):
        lastS1 = lastS1 + covariances[i]
        lastS2 = lastS2 - covariances[i]

        sMatrixList.append((lastS1, lastS2))

        s1 = (1./float(i+1)) * lastS1
        s2 = (1./float(n-i)) * lastS2

        s1Det = np.linalg.det(s1)
        s2Det = np.linalg.det(s2)

        v = float(i) / float(n)

        ratio = sFullDet / (np.power(s1Det, v) * np.power(s2Det, 1-v))

        likelihoodRatioStats[i] = n * math.log(ratio)

    maxIndex = np.argmax(likelihoodRatioStats[d:n-d]) + d
    maxStat = likelihoodRatioStats[maxIndex]

    logger.debug("Found max %f occurs at: %d", maxStat, index[maxIndex])
    
    criticalValue = critValueSim.getCriticalValue((k*(k+1)/2), alpha)
    if ( maxStat > criticalValue ):
        logger.debug("Found a potential change point at: %d", index[maxIndex])
        
        leftIndex = index[:maxIndex]
        rightIndex = index[maxIndex:]

        leftResiduals = residuals[:maxIndex]
        rightResidualsOrig = residuals[maxIndex:]

        # Calculate \Sigma and \Omega
        s1Matrix, s2Matrix = sMatrixList[maxIndex]
        sigma = 1./float(maxIndex) * s1Matrix
        omega = 1./float(n - maxIndex) * s2Matrix

        ellSigma = np.linalg.cholesky(sigma)
        ellSigmaInv = np.linalg.inv(ellSigma)
        ellOmega = np.linalg.cholesky(omega)

        wMatrix = np.dot(ellOmega, ellSigmaInv) - np.eye(k)

        # Construct a corrected series of residuals
        rightN = len(rightResidualsOrig)
        rightResiduals = []
        wMatrixUpdate = np.linalg.inv(np.eye(k) + wMatrix)
        for i in range(rightN):
            residual = residuals[maxIndex + i]
            newResidual = np.dot(wMatrixUpdate, residual)
            rightResiduals.append(newResidual)
        rightResiduals = np.array(rightResiduals)

        results[index[maxIndex]] = wMatrix

        if ( len(leftResiduals) >= 2*d + 1 ):
            recursiveChangePointDetector(leftResiduals, p, q, k, leftIndex, results, alpha)
            
        if ( len(rightResiduals) >= 2*d + 1 ):
            recursiveChangePointDetector(rightResiduals, p, q, k, rightIndex, results, alpha)
        
    return results

def changePointDetectorInit(dataframe, alpha=0.001):

    k = dataframe.shape[1]
    p = 1
    q = 0

    modeler = tsa.vector_ar.var_model.VAR(dataframe.as_matrix())
    model = modeler.fit()

    p = int(model.k_ar)
    logger.debug("P: %s", p.__str__())

    return recursiveChangePointDetector(model.resid, p, q, k, dataframe.index, {}, alpha)

if __name__ == '__main__':
    import sys

    if ( len(sys.argv) < 2 ):
    	print "Usage: %s <csv_data>" % sys.argv[0]
    	exit(1)

    alpha = 0.001
    if ( len(sys.argv) > 2 ):
        alpha = float(sys.argv[2])

    dataPath = sys.argv[1]
    df = pandas.read_csv(dataPath)
    print "Finished reading data."

    if ( type(df[df.columns[0]][0]) == str ):
        print "Reindexing using column:", df.columns[0]
        df['index'] = pandas.DatetimeIndex(df[df.columns[0]])
        df = df.set_index('index')
        df = df[df.columns[1:]]
        df = df.sort_index()

    results = changePointDetectorInit(df, alpha)
    print sorted(results.keys())
    print "Number of Change Points:", len(results.keys())
