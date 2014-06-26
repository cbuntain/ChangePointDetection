#!/usr/bin/python

import numpy as np

criticalValueMap = {}

def simulateCriticalValue(degreeFreedom, runs):

	# Do stuff
	maxList = []
	for i in range(runs):
		chiSqData = [np.random.chisquare(degreeFreedom) for x in range(10000)]
		maxList.append(max(chiSqData))

	return sorted(maxList)

def getCriticalValue(degreeFreedom, alpha=0.05, runs=10000):

	lambdaMaxDistribution = None
	lambdaMaxDistributionLen = None
	if ( not degreeFreedom in criticalValueMap ):
		lambdaMaxDistribution = simulateCriticalValue(degreeFreedom, runs)
		lambdaMaxDistributionLen = runs

		criticalValueMap[degreeFreedom] = (lambdaMaxDistribution, runs)

	else:
		(lambdaMaxDistribution, lambdaMaxDistributionLen) = criticalValueMap[degreeFreedom]

	critIndex = int(lambdaMaxDistributionLen * alpha)
	
	return lambdaMaxDistribution[critIndex]


if __name__ == "__main__":

	import sys

	if ( len(sys.argv) < 2 ):
		print("Usage: %s <df> [alpha] [runs]" % sys.argv[0])
		exit(1)

	degreeFreedom = float(sys.argv[1])
	alpha = 0.05
	runs = 100

	if ( len(sys.argv) > 2 ):
		alpha = float(sys.argv[2])

	if ( len(sys.argv) > 3 ):
		runs = int(sys.argv[3])

	critVal = getCriticalValue(degreeFreedom, alpha, runs)

	print("Critical Value at alpha=%f with DF=%f: %f" % (alpha, degreeFreedom, critVal))


