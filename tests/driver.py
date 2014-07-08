#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd

import scipy.linalg

from simulate_data import simulateVAR

sys.path.append(os.path.join("..", "lrt"))
sys.path.append(os.path.join("..", "cusum"))

from lrtPointDetectorWithW import changePointDetectorInit
from cusum_change_detector import cusum_algorithm

def cusumWrapper(data):

	changepoints = cusum_algorithm(data.as_matrix(), 1.36)

	mappedPts = dict(zip(changepoints, tuple([None]*len(changepoints))))

	return mappedPts

lrtDetector = lambda x: changePointDetectorInit(x, alpha=0.05)
cusumDetector = cusumWrapper

n = 1000
runs = 100

actualCp = []
foundCpLrt = []
foundCpCusum = []

for i in range(runs):

	k = np.random.random_integers(2,5)
	numChanges = np.random.random_integers(0,9)

	Phi = None
	mean = None
	data = None
	covariances = None
	wMatrices = None
	changePts = None

	while (True):
		# Generate the data (currently just VAR(1) process).
		Phi = np.random.randn(k,k)
		(w, v) = np.linalg.eig(Phi)

		flag = True
		for eig in w:
			if ( np.isreal(eig) == False or (1.0/np.abs(eig)) <= 1 or (1.0/np.abs(eig)) > 10 ):
				flag = False
				break

		if ( flag == False ):
			continue
		
		print "Phi:\n", Phi
		print "1 / Phi Eigs:", 1.0/np.abs(w)
		print "|Phi|:", np.linalg.det(Phi)

		mean = np.random.random_integers(1000, size=(k,))

		(data, covariances, wMatrices, changePts) = simulateVAR(n, numChanges, Phi, mean)

		if ( np.max(np.isnan(data)) == 0 and np.max(np.isinf(data)) == 0 ):
			break
		else:
			print "Regenerating Phi and data..."

	print "Number of Change Pts:", numChanges
	print "Mean:\n", mean
	print "Data:\n", data
	print "Covariances:\n", covariances
	print "W Matrices:\n", wMatrices
	print "Change Points:\n", changePts

	df = pd.DataFrame(data)

	for (detector, accArr, index) in [(cusumDetector, foundCpCusum, "CUSUM"), (lrtDetector, foundCpLrt, "LRT")]:
		detectedPoints = detector(df)

		tp = 0
		fp = 0
		fn = 0

		actualChangePtsCopy = changePts[:]

		print "Type:", index
		for cp in sorted(detectedPoints.keys()):
			print "Change Point @", cp
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

	actualCp.append(numChanges)

actuals = np.sum(actualCp)
print "Actual Count:", actuals

for (accArr, alg) in [(foundCpLrt, "LRT"), (foundCpCusum, "CUSUM")]:
	truePos = np.sum([x[0] for x in accArr])
	falsePos = np.sum([x[1] for x in accArr])
	falseNeg = actuals - truePos

	print alg, "Found Count:", truePos
	print alg, "False Pos Count:", falsePos
	print alg, "Accuracy:", float(truePos) / float(actuals)

	precision = float(truePos) / float(truePos + falsePos)
	recall = float(truePos) / float(truePos + falseNeg)
	f1 = 2.0 * precision * recall / (precision + recall)

	print alg, "F1:", f1