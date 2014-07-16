#!/usr/bin/python

import os
import sys
import time
import json
import logging
import scipy.sparse
import scipy.linalg
import numpy as np
import pandas as pd
import scipy.io

from multiprocessing.pool import ThreadPool

sys.path.append(os.path.join("..", "lrt"))
sys.path.append(os.path.join("..", "cusum"))
sys.path.append(os.path.join("..", "kcd"))

from lrtPointDetectorWithW import changePointDetectorInit
from cusum_change_detector import cusum_algorithm
from kcd import kernelChangeDetection

def cusumWrapper(data):

	# 1.36 is the critical value for alpha = 0.05
	changepoints = cusum_algorithm(data.as_matrix(), 1.36)

	mappedPts = dict(zip(changepoints, tuple([None]*len(changepoints))))

	return mappedPts

def lrtWrapper(data):

	resultMap = changePointDetectorInit(data, alpha=0.05)

	return resultMap["points"]

def kcdWrapper(data, d):

	(changePoints, stats) = kernelChangeDetection(data, eta=0.55, d=d, gamma = 0.25, nu=0.125)

	mappedPts = dict(zip(changePoints, tuple([None]*len(changePoints))))

	return mappedPts

if ( len(sys.argv) < 2 ):
	print "Usage: %s input.mat" % sys.argv[0]
	exit(1)

# Set up logging
currentTime = time.time()
logger = logging.getLogger('driver')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
logFileName = "run.%f.log" % currentTime
fh = logging.FileHandler(logFileName)
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

# Set up the output files
resultsFileName = "run.%f.json" % currentTime
errFileName = "run.%f.err" % currentTime
errFile = open(errFileName, "w")



# Read in the matlab data
realizations = 5000

inputMatlabFile = sys.argv[1]
inputMatlabData = scipy.io.loadmat(inputMatlabFile)
matlabData = inputMatlabData['simData']

k = matlabData.shape[1] / realizations
changePointLocation = matlabData.shape[0] / 2

# Set up the detection functions
lrtDetector = lrtWrapper
kcdDetector = lambda x: kcdWrapper(x, matlabData.shape[0]/10)
cusumDetector = cusumWrapper

# Arrays for recording accuracy
actualCp = []
foundCpLrt = []
foundCpCusum = []
foundCpKcd = []

logger.info("k: %d", k)

# Thread pool for running algorithms
pool = ThreadPool()

for i in range(realizations):

	logger.info("Run: %i", i)

	changePts = [changePointLocation]
	actualCp.append(len(changePts))
	logger.info("Acutal Change Points: %s", changePts.__str__())

	df = pd.DataFrame(matlabData[:,i:i+k])

	cusumThread = pool.apply_async(cusumDetector, (df, ))
	lrtThread = pool.apply_async(lrtDetector, (df, ))
	kcdThread = pool.apply_async(kcdDetector, (df, ))

	algoList = [(cusumThread, foundCpCusum, "CUSUM"), \
				(lrtThread, foundCpLrt, "LRT"),\
				(kcdThread, foundCpKcd, "KCD")]

	for (detectThread, accArr, index) in algoList:
		
		logger.info("Algorithm: %s", index)

		# Perform detection and catch errors
		try:
			# detectedPoints = detector(df)
			detectedPoints = detectThread.get()
		except Exception as e:
			errorStr = "%s -- Run %d: %s Error: %s" % (time.strftime("%d %b %Y %H:%M:%S"), i, index, e.__str__())
			logger.error(errorStr)
			errFile.write(errorStr + "\n")

			raise

			continue

		tp = 0
		fp = 0
		fn = 0

		actualChangePtsCopy = changePts[:]

		foundChangePoints = sorted(detectedPoints.keys())
		logger.info("Change Points: %s", foundChangePoints.__str__())
		for cp in foundChangePoints:
			# print "W-Matrix:\n", detectedPoints[cp]

			flag = False
			ptRange = 20
			for realChangePt in changePts:
				if ( cp >= realChangePt - ptRange and cp <= realChangePt + ptRange ):
					
					flag = True

					if ( realChangePt in actualChangePtsCopy ):
						tp += 1
						actualChangePtsCopy.remove(realChangePt)
					#else:
						# Do nothing

			if ( flag == False ):
				fp += 1

		fn = max(0, len(changePts) - tp)

		accArr.append((tp, fp))

actuals = np.sum(actualCp)
logger.info("Actual Count: %d", actuals)

runResults = {"actuals": actuals}

for (accArr, alg) in [(foundCpLrt, "LRT"), (foundCpCusum, "CUSUM"), (foundCpKcd, "KCD")]:
	truePos = np.sum([x[0] for x in accArr])
	falsePos = np.sum([x[1] for x in accArr])
	falseNeg = actuals - truePos
	accuracy = float(truePos) / float(actuals)

	logger.info("%s - Found Count: %d", alg, truePos)
	logger.info("%s - False Pos Count: %d", alg, falsePos)
	logger.info("%s - Accuracy: %f", alg, accuracy)

	if ( truePos + falsePos == 0 ):
		continue

	precision = float(truePos) / float(truePos + falsePos)
	recall = float(truePos) / float(truePos + falseNeg)
	f1 = 2.0 * precision * recall / (precision + recall)

	logger.info("%s - F1: %f", alg, f1)

errFile.close()