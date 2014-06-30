import numpy as np
import matplotlib.pylab as plt
import csv

num_dimensions = 3
total_time = 1000
num_changepoints = 2
 
data = []
covariances = []
for i in range(num_changepoints + 1):
    n = int(total_time / (num_changepoints + 1))

    # Generate a symmetric, positive semidefinite matrix.
    M = np.random.rand(num_dimensions, num_dimensions)
    M = M + M.T + num_dimensions * np.identity(num_dimensions)
    M = 3**i * M # Increase the covariance after each changepoint.
    np.linalg.cholesky(M) # Check that M is symmetric positive semidefinite.

    # Generate n innovations.
    innovation_covariance = M
    innovation_mean = np.array([0] * num_dimensions)
    innovations = np.random.multivariate_normal(innovation_mean,
                                                innovation_covariance, n)

    # Start the process with a draw from the distribution of innovations.
    if data == []:
        data.extend(np.random.multivariate_normal(innovation_mean,
                                                  innovation_covariance, 1))

    # Generate the data (currently just VAR(1) process).
    Phi = np.array([[0.6, 0.2, 0],
                    [0.2, 0.4, 0],
                    [0.6, 0.2, 0.5]])
    for t in range(n):
        datapoint = np.dot(Phi, data[(i*n) + t]) + innovations[t]
        data.append(datapoint.tolist())
    covariances.append(innovation_covariance)

'''
# Look at the time series of the first component.
first_component = [ datapoint[0] for datapoint in data ]
plt.plot(first_component)
plt.show()
'''

# Write the data.
data_output = 'simulation_{}dim_{}changes_data.csv'.format(
    num_dimensions, num_changepoints)
with open(data_output, 'w') as fp:
    writer = csv.writer(fp, delimiter=',')
    writer.writerows(data)

# Write the covariance matrices.
covariance_output = 'simulation_{}dim_{}changes_covariances.csv'.format(
    num_dimensions, num_changepoints)
with open(covariance_output, 'w') as fp:
    for i in range(len(covariances)):
        fp.write('Omega {}:\n'.format(i))
        fp.write(str(covariances[i]))
        fp.write('\n\n')

        # Also compute and write W after the first changepoint.
        if i != 0:
            L_Sigma = np.linalg.cholesky(covariances[i-1])
            L_Omega = np.linalg.cholesky(covariances[i])
            W = L_Omega * np.linalg.inv(L_Sigma) - np.identity(num_dimensions)
            fp.write('W {}:\n'.format(i))
            fp.write(str(W))
            fp.write('\n\n')
