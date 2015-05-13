import numpy as np

def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced 
    # predictions.
    # 
   
    mean_data = np.mean(data)
    ##Solution 1
    r_squared = 1.0 - np.square(data - predictions).sum()/np.square(data - mean_data).sum()
    
    ##Solution 2
    #r_squared = 1.0 - np.sum((data - predictions(**2)/np.sum((data - mean_data)**2)

    return r_squared
