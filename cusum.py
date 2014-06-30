import numpy as np
import statsmodels.tsa.api as tsa

# Import the data.
input_filename = 'simulated_data/simulation_3dim_2changes_data.csv'
with open(input_filename) as fp:
    data = np.array([ [ float(num) for num in line.split(',') ]
                      for line in fp ])

num_dimensions = len(data[0])
lag_order = 1
model = tsa.VAR(data)
results = model.fit(lag_order)

Phi = results.params[1: ]
residuals = [ np.dot(Phi, data[i]) - data[i+1] for i in range(len(data)-1) ]

d = (num_dimensions * (lag_order + 0 + 1)
     + (num_dimensions * (num_dimensions + 1)) / 2 + 1)

'''
print('penultimate datapoint: {}'.format(data[-2]))
print('last datapoint: {}'.format(data[-1]))
print('their forecast: {}'.format(results.forecast(data[-2:], 1)))
print('my forecast: {}'.format(np.dot(results.params[1:],
                                      np.transpose(data[-2]))))
print('params: {}'.format(results.params))
print('last epsilon: {}'.format(data[-1]-data[-2]))
'''
