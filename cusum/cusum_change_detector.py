#!/usr/bin/python3

import numpy as np
import statsmodels.tsa.api as tsa
import matplotlib.pylab as plt
import pandas
import sys



# Test statistic for a changepoint at h.
def C(h, square_residuals, num_dimensions):
    normalization = (h+1) / np.sqrt(2 * num_dimensions
                                    * len(square_residuals))
    A_h = sum(square_residuals[ :h+1])
    A_n = sum(square_residuals)
    centering = A_h / (h+1) - A_n / len(square_residuals)
    return normalization * centering

# Test statistic for a changepoint in [left, right] (inclusive).
def max_Cs(left, right, residuals, num_dimensions):
    resid = residuals[left:(right+1)]
    
    # Estimate the covariance matrix of the innovations.
    Sigma = sum([ e * e.T for e in resid ]) / (len(resid) - 1)
    Sigma_inv = Sigma.I

    square_residuals = [ float(e.T * Sigma_inv * e) for e in resid ]

    Cs = [ np.abs(C(h, square_residuals, num_dimensions))
           for h in range(right - left + 1) ]

    max_value = 0
    max_index = 0
    for i, c in enumerate(Cs):
        if c > max_value:
            max_value = c
            max_index = i
    Gamma_max = max_value
    h_max = max_index + left

    return Gamma_max, h_max, Cs

# Critical value for distribution of Gamma_max based on Brownian bridge.
# P(sup|M| < a) = 1 + 2 \sum_{i=1}^\infty (-1)^i exp(-2 i^2 a^2)
def critical_value(alpha):
    precision = 0.001

    # Compute the maximum i for which the summand above is zero.
    max_i = 0
    while np.exp(-2 * max_i**2 * precision**2) != 0:
        max_i = max_i + 1

    # Use the cdf above to increase a until it hits the critical value.
    probability = 0
    a = 0
    while probability < (1-alpha):
        a = a + precision
        probability = 1 + 2 * sum([ (-1)**i * np.exp(-2 * i**2 * a**2) 
                                    for i in range(1, max_i+1) ])
    return a

# Follows the cusum procedure described by Galeano and Pena.
def cusum_algorithm(data, critical_value):

    num_dimensions = len(data[0])
    total_time = len(data)



    ################
    #### STEP 1 ####
    ################

    # Estimate a VAR(p) model and compute the residuals.
    model = tsa.VAR(data)
    results = model.fit()
    residuals = [ np.matrix(e).T for e in list(results.resid) ]
    d = int(num_dimensions * (results.k_ar + 0 + 1)
            + (num_dimensions * (num_dimensions + 1)) / 2 + 1)

    possible_changepoints = [0, total_time-1]

    # Initialize h_first and h_last to the endpoints +/- d.
    h_first = d
    h_last = total_time-1 - d

    while True:
        ################
        #### STEP 2 ####
        ################

        # Find the most likely changepoint (if any) between h_first and h_last.
        Gamma_max, h_max, Cs = max_Cs(h_first, h_last,
                                      residuals, num_dimensions)
        '''
        plt.plot(range(h_first, h_last+1), Cs)
        plt.axhline(critical_value, color='red')
        plt.axvline(h_max, color='gray', linestyle='--')
        plt.show()
        '''

        # If there are none between h_first and h_last, skip to step 4.
        if Gamma_max < critical_value:
            break

        # Otherwise, look for more changepoints.
        else:
            old_Gamma_max = Gamma_max
            old_h_max = h_max

            #################
            #### STEP 3a ####
            #################

            # Find the leftmost possible changepoint (i.e., leftmost point
            # with a significant test statistic). Make it the new h_first.
            while Gamma_max > critical_value:
                t_2 = h_max - 1
                Gamma_max, h_max, Cs = max_Cs(h_first, t_2,
                                              residuals, num_dimensions)
            h_first = t_2

            #################
            #### STEP 3b ####
            #################

            # Find the rightmost possible changepoint (i.e., rightmost point
            # with a significant test statistic). Make it the new h_last.
            Gamma_max = old_Gamma_max
            h_max = old_h_max
            while Gamma_max > critical_value:
                t_1 = h_max + 1
                Gamma_max, h_max, Cs = max_Cs(t_1, h_last,
                                              residuals, num_dimensions)
            h_last = t_1

            #################
            #### STEP 3c ####
            #################

            # If the time between h_first and h_last is higher our resolution d,
            # record them and repeat steps 2 and 3 in narrower interval.
            if np.abs(h_last - h_first) > d:
                possible_changepoints.append(h_first)
                possible_changepoints.append(h_last)

                h_first = h_first + d
                h_last = h_last - d
            # Otherwise, record the most likely changepoint from before and then
            # go to step 4.
            else:
                possible_changepoints.append(old_h_max)
                break


    ################
    #### STEP 4 ####
    ################

    possible_changepoints.sort()

    # Delete possible changepoints until convergence.
    converged = False
    while not converged:
        # For every ith and (i+2)th changepoint, check if the open interval
        # between them is statistically significant. If not, drop the (i+1)th.
        for i in range(len(possible_changepoints)-2):
            Gamma_max, h_max, Cs = max_Cs(possible_changepoints[i]+1,
                                          possible_changepoints[i+2]-1,
                                          residuals, num_dimensions)

            if Gamma_max < critical_value:
                # Mark for deletion.
                possible_changepoints[i+1] = -1

        converged = True
        # Delete the marked ones.
        for i in reversed(range(len(possible_changepoints))):
            if possible_changepoints[i] == -1:
                del possible_changepoints[i]
                converged = False

    # Also delete the endpoints.
    del possible_changepoints[0]
    del possible_changepoints[-1]

    changepoints = [ point + 1 for point in possible_changepoints ]
    return tuple(changepoints)


if __name__ == '__main__':
    # Import the data.
    input_filename = sys.argv[1]
    with open(input_filename) as fp:
        if 'simulation' in input_filename:
            data = [ np.matrix([ float(num) for num in line.split(',') ]).T
                     for line in fp ]
        elif 'full.csv' in input_filename:
            df = pandas.read_csv(input_filename)
            if ( type(df[df.columns[0]][0]) == str ):
                df['index'] = pandas.DatetimeIndex(df[df.columns[0]])
                df = df.set_index('index')
                df = df[df.columns[1:]]
                df = df.sort_index()
            data = df.as_matrix()


    #critical_value = critical_value(0.000001)
    #print(critical_value)
    #critical_value = 1.36 # 95%
    #critical_value = 1.628 # 99%
    #critical_value = 1.95 # 99.9%
    #critical_value = 2.226 # 99.99%
    #critical_value = 2.471 # 99.999%
    critical_value = 2.694 # 99.9999%

    changepoints = cusum_algorithm(data, critical_value)

    print(changepoints)
