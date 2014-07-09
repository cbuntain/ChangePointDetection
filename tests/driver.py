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

from multiprocessing.pool import ThreadPool

from simulate_data import simulateVAR

sys.path.append(os.path.join("..", "lrt"))
sys.path.append(os.path.join("..", "cusum"))

from lrtPointDetectorWithW import changePointDetectorInit
from cusum_change_detector import cusum_algorithm

def cusumWrapper(data):

	# 1.36 is the critical value for alpha = 0.05
	changepoints = cusum_algorithm(data.as_matrix(), 1.36)

	mappedPts = dict(zip(changepoints, tuple([None]*len(changepoints))))

	return mappedPts

lrtDetector = lambda x: changePointDetectorInit(x, alpha=0.05)
cusumDetector = cusumWrapper

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


# Tunable parameters for testing
n = 10000
runs = 100
kRange = (1,6)
cpRange = (0,10)

logger.info("n: %d", n)
logger.info("Runs: %d", runs)
logger.info("k Range: %s", kRange.__str__())
logger.info("Change Point Count Range: %s", cpRange.__str__())

# Set up the output files
resultsFileName = "run.%f.json" % currentTime
errFileName = "run.%f.err" % currentTime
errFile = open(errFileName, "w")

# Arrays for recording accuracy
actualCp = []
foundCpLrt = []
foundCpCusum = []

# Array for results
results = []

# Thread pool for running algorithms
pool = ThreadPool(processes=2)

for i in range(runs):

	paramMap = {}

	k = 2**np.random.random_integers(kRange[0], kRange[1])
	numChanges = np.random.random_integers(cpRange[0], cpRange[1])

	logger.info("Run: %d", i)
	logger.info("k: %d", k)
	logger.info("Number of Change Pts: %d", numChanges)

	Phi = None
	mean = None
	data = None
	covariances = None
	wMatrices = None
	changePts = None

	logger.info("About to generate new Phi and data...")
	while (True):
		# Generate the data (currently just VAR(1) process).
		#Phi = np.random.randn(k,k)
		Phi = scipy.sparse.rand(k, k, density=0.25).todense()
		Phi = Phi + Phi.T + np.random.random_integers(2,9) * np.eye(k)
		(w, v) = np.linalg.eig(Phi)

		maxLambda = np.max(w)
		if ( maxLambda >= 1 ):
			Phi = Phi / (maxLambda + 1.0)
			(w, v) = np.linalg.eig(Phi)

		flag = True
		msg = None
		if ( np.sum(np.isreal(w) != True) > 0 ):
			flag = False
			msg = "Complex eigenvalues"
		if ( np.min(1.0/np.abs(w)) <= 1 ):
			flag = False
			msg = "On/Inside the unit circle"
		# if ( np.max(1.0/np.abs(w)) > 100 ) :
		# 	flag = False
		# 	msg = "Inverted eigenvalue is too large"

		if ( flag == False ):
			logger.info("Poor eig construct [%s]: [%s] - Regenerating Phi and data...", msg, w.__str__())
			continue
		
		mean = np.random.random_integers(1000, size=(k,))

		(data, covariances, wMatrices, changePts) = simulateVAR(n, numChanges, Phi, mean)

		if ( np.sum(np.isnan(data)) == 0 and np.sum(np.isinf(data)) == 0 ):
			break
		else:
			logger.info("Poor data. Regenerating Phi and data...")

	logger.info("Mean: %s", mean.__str__())
	logger.info("Change Points: %s", changePts.__str__())
	logger.debug("Data:\n%s", data.__str__())
	logger.debug("Covariances:\n%s", covariances.__str__())
	logger.debug("W Matrices:\n%s", wMatrices.__str__())
	logger.debug("Phi:\n%s", Phi.__str__())
	logger.debug("1 / |Phi Eigs|: %s", (1.0/np.abs(w)).__str__())

	paramMap["run"] = i
	paramMap["k"] = k
	paramMap["changePointCount"] = numChanges
	paramMap["changePoints"] = changePts
	paramMap["mean"] = mean.tolist()
	paramMap["phi"] = Phi.tolist()
	paramMap["cov"] = covariances.tolist()

	df = pd.DataFrame(data)


	cusumThread = pool.apply_async(cusumDetector, (df, ))
	lrtThread = pool.apply_async(lrtDetector, (df, ))

	for (detectThread, accArr, index) in [(cusumThread, foundCpCusum, "CUSUM"), (lrtThread, foundCpLrt, "LRT")]:
		
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

		# Update the param map with results
		paramMap[index] = {"points": foundChangePoints, "tp": tp, "fp": fp, "fn": fn}

		accArr.append((tp, fp))

	actualCp.append(numChanges)

	# Save the results from this run
	results.append(paramMap)

actuals = np.sum(actualCp)
logger.info("Actual Count: %d", actuals)

runResults = {"actuals": actuals}

for (accArr, alg) in [(foundCpLrt, "LRT"), (foundCpCusum, "CUSUM")]:
	truePos = np.sum([x[0] for x in accArr])
	falsePos = np.sum([x[1] for x in accArr])
	falseNeg = actuals - truePos
	accuracy = float(truePos) / float(actuals)

	logger.info("%s - Found Count: %d", alg, truePos)
	logger.info("%s - False Pos Count: %d", alg, falsePos)
	logger.info("%s - Accuracy: %f", alg, accuracy)

	precision = float(truePos) / float(truePos + falsePos)
	recall = float(truePos) / float(truePos + falseNeg)
	f1 = 2.0 * precision * recall / (precision + recall)

	logger.info("%s - F1: %d", alg, f1)

	runResults[alg] = {"tp": truePos, "fp": falsePos, "acc": accuracy, "precision": precision, "recall": recall, "f1": f1}

results.append(runResults)

resultsFile = open(resultsFileName, "w")
resultsFile.write(json.dumps(results, indent=True))
resultsFile.close()

errFile.close()