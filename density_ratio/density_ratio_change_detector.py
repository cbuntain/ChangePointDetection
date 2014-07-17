#!/usr/bin/python3

# The following code is based on Kawahara and Sugiyama's paper,
# "Sequential Change-Point Detection Based on Direct
# Density-Ratio Estimation".

import numpy as np
import pandas
import sys
import matplotlib.pylab as plt



# Gaussian kernel.
def kernel(Y1, Y2, sigma):
    norm_sq = sum([ (Y1[i] - Y2[i])**2 for i in range(len(Y1)) ])
    return np.exp(- norm_sq / (2 * sigma**2))



# Estimate the Kullback-Liebler importance.
# (Denoted Algorithm 1 in the paper.)
def KLIEP_algorithm(ref_set, test_set, sigma, ratio_args):
    print('Beginning KLIEP algorithm')
    K = [ [ kernel(test_set[i], test_set[l], sigma)
            for l in range(len(test_set)) ]
          for i in range(len(test_set)) ]
    K = np.matrix(K)

    b = [ 1 / len(ref_set) * sum([ kernel(ref_set[i], test_set[l], sigma)
                                   for i in range(len(ref_set)) ])
          for l in range(len(test_set)) ]
    b = np.matrix(b).T

    alpha = np.matrix([1] * len(test_set)).T

    # Perform gradient ascent.
    learning_rate = 0.001 #0.001
    zeros = np.matrix([0] * len(alpha)).T
    old_alpha = alpha
    while True:
        alpha = alpha + learning_rate * K * (1/(K * alpha))
        alpha = alpha + (1 - b.T * alpha)[0,0] * b / (b.T * b)
        alpha = np.maximum(alpha, zeros)
        alpha = alpha / (b.T * alpha)

        # Check for L1 convergence.
        norm = sum([ np.abs(alpha[i] - old_alpha[i])
                     for i in range(len(alpha)) ])
        if norm < 0.01:#0.001:
            print('Converged')
            break

        old_alpha = alpha

    ratios = [ sum([ alpha[l] * kernel(ratio_args[i], test_set[l], sigma)
                     for l in range(len(test_set)) ])
               for i in range(len(ratio_args)) ]
    return alpha, ratios

# Use cross-validation to select sigma of Gaussian kernel.
# (Denoted Algorithm 2 in the paper.)
def select_sigma(ref_set, test_set, sigmas):
    print('Beginning selection of sigma')
    # Cut the test set into 5 folds of roughly equal length.
    num_folds = 5
    len_fold = int(len(test_set) / num_folds)
    folds = [ test_set[(i*len_fold) : ((i+1)*len_fold)]
              for i in range(num_folds) ]
    len_leftover = len(test_set) - len_fold * num_folds
    if len_leftover != 0:
        folds[-1].extend(test_set[-len_leftover: ])

    avg_loglihoods = []
    for sigma in sigmas:
        print('Trying sigma={}'.format(sigma))
        loglihoods = []

        # Use the current sigma and the rest of the folds to compute
        # the log likelihood of the holdout fold.
        for r in range(num_folds):
            other_folds = []
            [ other_folds.extend(folds[i]) for i in range(num_folds)
              if i != r ] # Flatten the list of lists.
            ratios, alpha = KLIEP_algorithm(ref_set, other_folds,
                                            sigma, folds[r])
            loglihood = sum([ np.log(r) for r in ratios ]) / len(folds[r])
            loglihoods.append(loglihood)

        avg_loglihood = sum(loglihoods) / num_folds
        avg_loglihoods.append(avg_loglihood)
    
    i = avg_loglihoods.index(max(avg_loglihoods))

    print('Selected sigma={}'.format(sigmas[i]))
    return sigmas[i]


        
# (Denoted Algorithm 3 in the paper.)
def update_alpha(new_datapoint, alpha, eta, lambdaa,
                 ref_set, test_set, sigma):
    # Add new datapoint to test set.
    new_Y = test_set[-1][len(new_datapoint): ]
    new_Y = np.append(new_Y, new_datapoint)
    test_set.append(np.array(new_Y))

    # Update alpha.
    alpha = alpha.T.tolist()[0] # First flatten alpha to a plain old list.
    c = sum([ alpha[l] * kernel(test_set[-1], test_set[l], sigma)
              for l in range(len(test_set)-1) ])
    alpha = [ (1 - eta * lambdaa) * alpha[i+1]
              for i in range(len(alpha)-1) ]
    alpha.append(eta / c)
    alpha = np.matrix(alpha).T

    b = [ 1 / len(ref_set) * sum([ kernel(ref_set[i], test_set[l], sigma)
                                   for i in range(len(ref_set)) ])
          for l in range(1, len(test_set)) ]
    b = np.matrix(b).T

    zeros = np.matrix([0] * len(alpha)).T
    alpha = alpha + (1 - b.T * alpha)[0,0] * b / (b.T * b)
    alpha = np.maximum(alpha, zeros)
    alpha = alpha / (b.T * alpha)

    ref_set.append(test_set[0])
    del test_set[0]

    return alpha, ref_set, test_set



def get_ref_test_sets(data, start,
                      window_size, len_ref_set, len_test_set):
    ref_set = [ np.array(data[t : (t+window_size)]).flatten() 
                for t in range(start + len_ref_set) ]
    test_set = [ np.array(data[t : (t+window_size)]).flatten()
                 for t in range(start + len_ref_set,
                                start + len_ref_set + len_test_set) ]
    return ref_set, test_set



# (Denoted Algorithm 4 in the paper.)
def online_algorithm(data, window_size, len_ref_set, len_test_set,
                     sigmas, threshold, eta, lambdaa):
    changepoints = []

    # Cut data into reference set (before the possible change) and
    # test set (after the possible change).
    start = 0
    ref_set, test_set = get_ref_test_sets(data, start, window_size,
                                          len_ref_set, len_test_set)

    # Use cross-validation to find optimal sigma.
    sigma = select_sigma(ref_set, test_set, sigmas)

    scores = []

    done = False
    while not done:
        print('New online iteration')
        alpha, ratios = KLIEP_algorithm(ref_set, test_set, sigma, ref_set)

        t = start + len_ref_set + len_test_set + window_size
        while t != len(data):
            print('Beginning update of alpha: t={}'.format(t))
            alpha, ref_set, test_set = update_alpha(data[t], alpha,
                                                    eta, lambdaa,
                                                    ref_set, test_set, sigma)
            ratios = [ sum([ alpha[l] * kernel(test_set[i], test_set[l], sigma)
                             for l in range(len(test_set)) ])
                       for i in range(len(test_set)) ]
            log_ratios = [ np.log(r) for r in ratios ]
            score = sum(log_ratios)[0,0]
            scores.append(score)
            print('Score={}'.format(score))

            if score > threshold:
                changepoint.append(t)
                print('changepoint: {}'.format(t))
                
                # If we're out of data, abort.
                if t + len_ref_set + len_test_set > len(data):
                    done = True
                # Otherwise, move the reference and test sets to after the
                # changepoint. If this were actually online, it would require
                # waiting until the current time is
                # t + len_ref_set + len_test_set.
                else:
                    start = t
                    ref_set, test_set = get_ref_test_sets(data, start,
                                                          window_size,
                                                          len_ref_set,
                                                          len_test_set)
                    break
            else:
                t = t + 1
        # If the while loop ends without being broken, we're out of data.
        else: 
            done = True

    return changepoints, scores
            


if __name__ == '__main__':
    # Import the data.
    input_filename = sys.argv[1]
    with open(input_filename) as fp:
        if 'simulation' in input_filename:
            data = [ [ float(num) for num in line.split(',') ]
                     for line in fp ]
        elif 'full.csv' in input_filename:
            df = pandas.read_csv(input_filename)
            if ( type(df[df.columns[0]][0]) == str ):
                df['index'] = pandas.DatetimeIndex(df[df.columns[0]])
                df = df.set_index('index')
                df = df[df.columns[1:]]
                df = df.sort_index()

    # Normalize the data into [-1,1].
    # (Notably, this wouldn't be possible in an actually online setting.
    #  Perhaps use a logistic function instead.)
    num_dimensions = len(data[0])
    for d in range(num_dimensions):
        maxx = max([ np.abs(datapoint[d]) for datapoint in data ])
        for i in range(len(data)):
            data[i][d] = data[i][d] / maxx

    # The following parameter settings mostly follow page 11.
    window_size = 80 #80
    len_ref_set = 100 #100
    len_test_set = 100 #100
    threshold = 0.4
    eta = 1.0
    lambdaa =  0.01
    sigmas = list(np.arange(1, 3, 1))#list(range(1,11))

    changepoints = online_algorithm(data, window_size,
                                    len_ref_set, len_test_set,
                                    sigmas, threshold, eta, lambdaa)
    print('Final changepoints: {}'.format(changepoints))
    
    plt.plot(scores)
    plt.show()
