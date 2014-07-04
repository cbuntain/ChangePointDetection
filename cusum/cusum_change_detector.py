import numpy as np
import statsmodels.tsa.api as tsa
import matplotlib.pylab as plt



# Test statistic for a changepoint at h.
def C(h, square_residuals):
    normalization = (h+1) / np.sqrt(2 * num_dimensions
                                    * len(square_residuals))
    A_h = sum(square_residuals[ :h+1])
    A_n = sum(square_residuals)
    centering = A_h / (h+1) - A_n / len(square_residuals)
    return normalization * centering

# Test statistic for a changepoint in [left, right] (inclusive).
def max_Cs(left, right):
    resid = residuals[left:(right+1)]
    
    # Estimate the covariance matrix of the innovations.
    Sigma = sum([ e * e.T for e in resid ]) / (len(resid) - 1)
    Sigma_inv = Sigma.I

    square_residuals = [ float(e.T * Sigma_inv * e) for e in resid ]

    Cs = [ np.abs(C(h, square_residuals)) for h in range(right - left + 1) ]

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






# Import the data.
input_filename = '../simulated_data/simulation_3dim_9changes_data.csv'
with open(input_filename) as fp:
    data = [ np.matrix([ float(num) for num in line.split(',') ]).T
             for line in fp ]

num_dimensions = len(data[0])
lag_order = 1
d = int(num_dimensions * (lag_order + 0 + 1)
        + (num_dimensions * (num_dimensions + 1)) / 2 + 1)
total_time = len(data)

#critical_value = critical_value(0.000001)
#print(critical_value)
#critical_value = 1.36 # 95%
#critical_value = 1.628 # 99%
#critical_value = 1.95 # 99.9%
#critical_value = 2.226 # 99.99%
#critical_value = 2.471 # 99.999%
critical_value = 2.694 # 99.9999%


################
#### STEP 1 ####
################

# Estimate a VAR model and compute the residuals.
model = tsa.VAR(data)
results = model.fit(lag_order)
Phi = np.matrix(results.params[1: ])
intercept = np.matrix(results.params[0]).T
residuals = [ data[t+1] - (intercept + Phi * data[t])
              for t in range(total_time - 1) ]
residuals.insert(0, intercept)

possible_changepoints = [0, total_time-1]

# Initialize h_first and h_last to the endpoints +/- d.
h_first = d
h_last = total_time-1 - d

repeat_steps_2_and_3 = True
while repeat_steps_2_and_3:
    ################
    #### STEP 2 ####
    ################

    # Find the most likely changepoint (if any) between h_first and h_last.
    Gamma_max, h_max, Cs = max_Cs(h_first, h_last)
    '''
    plt.plot(range(h_first, h_last+1), Cs)
    plt.axhline(critical_value, color='red')
    plt.axvline(h_max, color='gray', linestyle='--')
    plt.show()
    '''

    # If there are none between h_first and h_last, skip to step 4.
    if Gamma_max < critical_value:
        repeat_steps_2_and_3 = False

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
            Gamma_max, h_max, Cs = max_Cs(h_first, t_2)
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
            Gamma_max, h_max, Cs = max_Cs(t_1, h_last)
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
            print('Possible changepoints: {}'.format(possible_changepoints))
        # Otherwise, record the most likely changepoint from before and then
        # go to step 4.
        else:
            possible_changepoints.append(old_h_max)
            print('Possible changepoints: {}'.format(possible_changepoints))
            repeat_steps_2_and_3 = False


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
                                      possible_changepoints[i+2]-1)

        if Gamma_max < critical_value:
            # Mark for deletion.
            possible_changepoints[i+1] = -1
            print('del')

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
print('Final changepoints: {}'.format(changepoints))
