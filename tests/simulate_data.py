import numpy as np

def simulateVAR(n, numChanges, phis, mean, covChange=lambda sigma, i: sigma, meanShift=lambda mu, i: mu):

    k = mean.shape[0]
    order = len(phis)

    data = np.zeros((n, k))
    covariances = np.zeros((numChanges + 1, k, k))

    # How many steps between each change point?
    windowSize = int(n / (numChanges + 1))

    # Find where all the change-points will occur
    changePoints = []
    for i in range(numChanges+1):
        changePoints.append(windowSize * i)
    changePoints.append(n)

    # Generate data for each change point
    for i in range(len(changePoints) - 1):
        start = changePoints[i]
        end = changePoints[i+1]

        # Generate a symmetric, positive semidefinite matrix.
        M = None
        tryCount = 100
        while ( tryCount > 0 ):
            tryCount -= 1

            try:
                M = np.random.rand(k, k)
                M = M + M.T + (np.random.randint(k)+k) * np.identity(k)
                # M = 3**(i+1) * M # Increase the covariance after each changepoint.
                M = covChange(M, i)
                np.linalg.cholesky(M) # Check that M is symmetric positive semidefinite.

                # If we get this far, we have a good M matrix
                break

            except:
                # Ignore and retry
                if ( tryCount == 0 ):
                    raise

        # Generate n innovations.
        innovation_covariance = M
        innovation_mean = np.array([0] * k)
        innovations = np.random.multivariate_normal(innovation_mean,
                                                    innovation_covariance, n)

        # Prime the process with initial values
        if ( start == 0 ):
            for t in range(order):
                newValue = np.random.multivariate_normal(np.zeros(k), innovation_covariance, 1)
                data[t] = newValue + innovations[t]
            start = order

        for t in range(start, end):
            datapoint = np.zeros((1, k))
            for phiIndex in range(order):
                phi = phis[phiIndex]
                laggedDataPointIndex = t - 1 - phiIndex
                laggedData = np.dot(phi, (data[laggedDataPointIndex]))
                datapoint += laggedData
            datapoint += innovations[t]
            data[t] = datapoint

        covariances[i] = innovation_covariance

    # Apply mean shifts
    for i in range(len(changePoints) - 1):
        start = changePoints[i]
        end = changePoints[i+1]

        data[start:end] += meanShift(mean, i)

    # Also compute and write W after the first changepoint.
    wMatrices = np.zeros((numChanges, k, k))
    for i in range(1, len(covariances)):

        L_Sigma = np.linalg.cholesky(covariances[i-1])
        L_Omega = np.linalg.cholesky(covariances[i])
        W = L_Omega * np.linalg.inv(L_Sigma) - np.identity(k)

        wMatrices[i-1] = W

    return (data, covariances, wMatrices, changePoints[1:-1])
