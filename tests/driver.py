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
sys.path.append(os.path.join("..", "kcd"))

from lrtPointDetectorWithW import changePointDetectorInit
from cusum_change_detector import cusum_algorithm
from kcd import kernelChangeDetection

def generateCandidateMatrix(k):
	
	candidate = None

	while (True):
		# Generate the data (currently just VAR(1) process).
		#Phi = np.random.randn(k,k)
		candidate = scipy.sparse.rand(k, k, density=0.25).todense()
		candidate = candidate + candidate.T + np.random.random_integers(2,9) * np.eye(k)
		(w, v) = np.linalg.eig(candidate)

		maxLambda = np.max(w)
		if ( maxLambda >= 1 ):
			candidate = candidate / (maxLambda + 1.0)
			(w, v) = np.linalg.eig(candidate)

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
			logger.info("Poor eig construct [%s]: [%s] - Regenerating candidate...", msg, w.__str__())
			continue
		else:
			break

	return candidate

def cusumWrapper(data):

	# 1.36 is the critical value for alpha = 0.05
	changepoints = cusum_algorithm(data.as_matrix(), 1.36)

	mappedPts = dict(zip(changepoints, tuple([None]*len(changepoints))))

	return mappedPts

def lrtWrapper(data):

	resultMap = changePointDetectorInit(data, alpha=0.05)

	return resultMap["points"]

def kcdWrapper(data):

	# These settings worked well for mean shifts
	(changePoints, stats) = kernelChangeDetection(data, d=200, eta=0.7, gamma=0.005, nu=0.001, useCov=True)

	# DO NOT UNCOMMENT THIS IF YOU RUN THIS FUNCTION AS A THREAD!1
	# import pylab as pl
	# pl.figure(figsize=(8, 6), dpi=80)
	# pl.plot(stats)
	# pl.legend()
	# pl.show()

	mappedPts = dict(zip(changePoints, tuple([None]*len(changePoints))))

	return mappedPts

lrtDetector = lrtWrapper
kcdDetector = kcdWrapper
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
# n = 1000
n = 500
runs = 10
#kRange = (2,12)
kRange = (2,2)
cpRange = (1,1)
#cpRange = (3,3)
varOrder = 1

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
foundCpKcd = []

# Array for results
results = []

# Thread pool for running algorithms
pool = ThreadPool()

for i in range(runs):

	paramMap = {}

	k = np.random.random_integers(kRange[0], kRange[1])
	numChanges = np.random.random_integers(cpRange[0], cpRange[1])

	logger.info("Run: %d", i)
	logger.info("k: %d", k)
	logger.info("Number of Change Pts: %d", numChanges)

	Phis = None
	mean = None
	data = None
	covariances = None
	wMatrices = None
	changePts = None

	logger.info("About to generate new Phi and data...")
	while ( True ):

		candidates = []
		for p in range(varOrder):
			candidates.append(generateCandidateMatrix(k))
		logger.info("Generate %d new candidates...", varOrder)

		PhiList = []
		if varOrder == 1:
			PhiList.append(candidates[0])
		elif varOrder == 2:
			PhiList.append(candidates[0] + candidates[1])
			PhiList.append(-1.0 * np.dot(candidates[0], candidates[1]))
		elif varOrder == 3:
			PhiList.append(candidates[0] + candidates[1] + candidates[2])
			PhiList.append(-1.0 * np.dot(candidates[0], candidates[1]) - \
				np.dot(candidates[1], candidates[2]) - \
				np.dot(candidates[0], candidates[2]))
			PhiList.append(np.dot(np.dot(candidates[0], candidates[1]), candidates[2]))
		else:
			print "Can't do beyond order 3, and you requested order:", varOrder
			exit(1)

		Phis = PhiList
		
		mean = np.random.random_integers(100, size=(k,))
		mean2 = np.random.random_integers(100, size=(k,))
		mean3 = np.random.random_integers(100, size=(k,))

		sigma = np.random.rand(k, k)
		sigma = sigma + sigma.T + (np.random.randint(k)+k) * np.identity(k)

		(data, covariances, wMatrices, changePts) = simulateVAR(n, numChanges, PhiList, mean, \
			covChange=lambda s, i: sigma, \
			meanShift=lambda mu, i: mu) #(i*10)*mu + mu

		logger.info("Mean: %s", mean.__str__())
		logger.info("Change Points: %s", changePts.__str__())
		logger.info("Data Length: %d", len(data))
		
		# import matplotlib.pyplot as plt
		# from mpl_toolkits.mplot3d import Axes3D
		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')
		# ax.plot(range(n), data[:,0], data[:,1])
		# plt.legend()
		# plt.show()

		if ( np.sum(np.isnan(data)) == 0 and np.sum(np.isinf(data)) == 0 ):
			break
		else:
			logger.info("Poor data. Regenerating Phi and data...")

	logger.info("Mean: %s", mean.__str__())
	logger.info("Change Points: %s", changePts.__str__())
	logger.debug("Data:\n%s", data.__str__())
	logger.debug("Covariances:\n%s", covariances.__str__())
	# logger.debug("W Matrices:\n%s", wMatrices.__str__())
	logger.debug("Phis:\n%s", Phis.__str__())
	# logger.debug("1 / |Phi Eigs|: %s", (1.0/np.abs(w)).__str__())

	paramMap["run"] = i
	paramMap["k"] = k
	paramMap["changePointCount"] = numChanges
	paramMap["changePoints"] = changePts
	# paramMap["mean"] = mean.tolist()
	# paramMap["phis"] = [x.tolist() for x in Phis]
	# paramMap["cov"] = covariances.tolist()

	df = pd.DataFrame(data)

	# kcdDetector(df)
	# exit(1)

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
			ptRange = 50
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

	runResults[alg] = {"tp": truePos, "fp": falsePos, "acc": accuracy, "precision": precision, "recall": recall, "f1": f1}

results.append(runResults)

resultsFile = open(resultsFileName, "w")
resultsFile.write(json.dumps(results, indent=True))
resultsFile.close()

errFile.close()
