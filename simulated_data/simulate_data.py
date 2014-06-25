import numpy as np
import matplotlib.pylab as plt
import csv

num_dimensions = 3
total_time = 1000
num_changepoints = 2
 
all_data = []
covariances = []
for i in range(num_changepoints + 1):
    # Add an extra datapoint to the last set so there are
    # total_time points overall.
    n = int(total_time / (num_changepoints + 1))
    if i == num_changepoints:
        n = n + 1

    # Generate a symmetric, positive semidefinite matrix.
    M = np.random.rand(num_dimensions, num_dimensions)
    M = M + np.transpose(M) + num_dimensions * np.identity(num_dimensions)
    M = M * 2**i
    np.linalg.cholesky(M) # Check that M is symmetric positive semidefinite.

    # Generate n errors.
    error_covariance = M
    error_mean = np.array([0] * num_dimensions)
    errors = np.random.multivariate_normal(error_mean, error_covariance, n)

    # Generate the data (currently just a white noise sequence).
    data = errors

    all_data.extend(data.tolist())
    covariances.append(error_covariance)

'''
# Look at the time series of the first component.
first_component = [ datapoint[0] for datapoint in all_data ]
plt.plot(first_component)
plt.show()
'''

# Write the data.
data_output = 'simulation_{}dim_{}changes_data.csv'.format(
    num_dimensions, num_changepoints)
with open(data_output, 'w') as fp:
    writer = csv.writer(fp, delimiter=',')
    writer.writerows(all_data)

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
