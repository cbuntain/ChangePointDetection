#!/usr/bin/python

import math
import pandas
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa as tsa
import pylab as pl

import critValueSim

def calculateS(residuals, offset, length):
    
    sMatrix = np.zeros((residuals.shape[1], residuals.shape[1]))
    
    for i in range(offset, offset+length):
        error = np.reshape(residuals[i], (residuals.shape[1], 1))
        errorCov = np.dot(error, error.T)

        sMatrix += errorCov

    sMatrix = (1./float(length)) * sMatrix
    
    return sMatrix

def recursiveChangePointDetector(residuals, p, q, k, index, results):

    d = k * (p + q + 1) + k + 1
    
    n = residuals.shape[0]

    # Pre-calculate covariances
    covariances = []
    for i in range(n):
    	error = np.reshape(residuals[i], (k, 1))
    	errorCov = np.dot(error, error.T)
    	covariances.append(errorCov)
    
    likelihoodRatioStats = np.zeros(n)
    sF = calculateS(residuals, 0, n)
    sFullDet = np.linalg.det(sF)

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

    testStat = np.max(likelihoodRatioStats[d:n-d])

    print "Found Max:", testStat, "Max:", maxStat, "occurs at:", maxIndex
    
    criticalValue = critValueSim.getCriticalValue((k*(k+1)/2))
    if ( maxStat > criticalValue ):
        print "Found a potential change point at:", index[maxIndex]
        
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
            recursiveChangePointDetector(leftResiduals, p, q, k, leftIndex, results)
            
        if ( len(rightResiduals) >= 2*d + 1 ):
            recursiveChangePointDetector(rightResiduals, p, q, k, rightIndex, results)
        
    return results

def changePointDetectorInit(dataframe, p, q):

    k = dataframe.shape[1]

    modeler = tsa.vector_ar.var_model.VAR(dataframe.as_matrix())
    model = modeler.fit()

    return recursiveChangePointDetector(model.resid, p, q, k, dataframe.index, {})

import sys

if ( len(sys.argv) < 2 ):
	print "Usage: %s <csv_data>" % sys.argv[0]
	exit(1)

dataPath = sys.argv[1]
df = pandas.read_csv(dataPath, header=None)
print "Finished reading data."

results = changePointDetectorInit(df, 1, 0)
print sorted(results.keys())