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

def recursiveChangePointDetector(dataframe, p, q, results, alpha=0.001):
    
    k = dataframe.shape[1]
    
    modeler = tsa.vector_ar.var_model.VAR(dataframe.as_matrix())
    model = modeler.fit()

    p = int(model.k_ar)
    q = 0
    d = k * (p + q + 1) + k + 1
    
    n = model.resid.shape[0]

    # Pre-calculate covariances
    covariances = []
    for i in range(n):
    	error = np.reshape(model.resid[i], (k, 1))
    	errorCov = np.dot(error, error.T)
    	covariances.append(errorCov)
    
    likelihoodRatioStats = np.zeros(n)
    sF = calculateS(model.resid, 0, n)
    sFullDet = np.linalg.det(sF)

    lastS1 = np.zeros((k,k))
    lastS2 = float(n) * sF

    for i in range(n):
        lastS1 = lastS1 + covariances[i]
        lastS2 = lastS2 - covariances[i]

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
    
    criticalValue = critValueSim.getCriticalValue((k*(k+1)/2), alpha)
    if ( maxStat > criticalValue ):
        print "Found a potential change point at:", dataframe.index[maxIndex]
        
        results[dataframe.index[maxIndex]] = model
        
        leftDataFrame = dataframe[:maxIndex]
        rightDataFrame = dataframe[maxIndex:]
        
        if ( leftDataFrame.shape[0] >= 2*d + 1 ):
            recursiveChangePointDetector(leftDataFrame, p, q, results, alpha)
            
        if ( rightDataFrame.shape[0] >= 2*d + 1 ):
            recursiveChangePointDetector(rightDataFrame, p, q, results, alpha)
        
    return results

import sys

if ( len(sys.argv) < 2 ):
	print "Usage: %s <csv_data> [alpha]" % sys.argv[0]
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

results = recursiveChangePointDetector(df, 1, 0, {}, alpha)
print sorted(results.keys())
print "Number of Change Points:", len(results.keys())