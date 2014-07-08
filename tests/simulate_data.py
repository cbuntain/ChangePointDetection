import numpy as np

def simulateVAR(n, numChanges, phi, mean):

    k = mean.shape[0]

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
                M = M + M.T + k * np.identity(k)
                M = 3**(i+1) * M # Increase the covariance after each changepoint.
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

        # Prime the process with an initial value
        if ( start == 0 ):
            firstValue = np.random.multivariate_normal(mean, innovation_covariance, 1)
            data[0] = firstValue + innovations[0]
            start = 1

        for t in range(start, end):
            datapoint = np.dot(phi, data[t-1]) + innovations[t]
            data[t] = datapoint
        covariances[i] = innovation_covariance

    # Also compute and write W after the first changepoint.
    wMatrices = np.zeros((numChanges, k, k))
    for i in range(1, len(covariances)):

        L_Sigma = np.linalg.cholesky(covariances[i-1])
        L_Omega = np.linalg.cholesky(covariances[i])
        W = L_Omega * np.linalg.inv(L_Sigma) - np.identity(k)

        wMatrices[i-1] = W

    return (data, covariances, wMatrices, changePoints[1:-1])
