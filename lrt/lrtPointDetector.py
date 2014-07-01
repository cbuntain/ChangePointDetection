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

def recursiveChangePointDetector(dataframe, p, q, results):
    
    k = dataframe.shape[1]
    d = k * (p + q + 1) + k + 1
    
    modeler = tsa.vector_ar.var_model.VAR(dataframe.as_matrix())
    model = modeler.fit()
    
    n = model.resid.shape[0]
    
    likelihoodRatioStats = np.zeros(n)
    sF = calculateS(model.resid, 0, n)
    sFullDet = np.linalg.det(sF)

    for i in range(d, n - d):
        s1 = calculateS(model.resid, 0, i)
        s2 = calculateS(model.resid, i, n-i)

        s1Det = np.linalg.det(s1)
        s2Det = np.linalg.det(s2)

        v = float(i) / float(n)

        ratio = sFullDet / (np.power(s1Det, v) * np.power(s2Det, 1-v))

        likelihoodRatioStats[i] = n * math.log(ratio)

    maxIndex = np.argmax(likelihoodRatioStats)
    maxStat = likelihoodRatioStats[maxIndex]

    print "Max:", maxStat, "occurs at:", maxIndex
    
    criticalValue = critValueSim.getCriticalValue((k*(k+1)/2))
    if ( maxStat > criticalValue ):
        print "Found a potential change point at:", dataframe.index[maxIndex]
        
        results[dataframe.index[maxIndex]] = model
        
        leftDataFrame = dataframe[:maxIndex]
        rightDataFrame = dataframe[maxIndex:]
        
        if ( leftDataFrame.shape[0] >= 2*d + 1 ):
            recursiveChangePointDetector(leftDataFrame, p, q, results)
            
        if ( rightDataFrame.shape[0] >= 2*d + 1 ):
            recursiveChangePointDetector(rightDataFrame, p, q, results)
        
    return results

import sys

if ( len(sys.argv) < 2 ):
	print "Usage: %s <csv_data>" % sys.argv[0]
	exit(1)

dataPath = sys.argv[1]
df = pandas.read_csv(dataPath, header=None)
print "Finished reading data."

results = recursiveChangePointDetector(df, 1, 0, {})
print sorted(results.keys())