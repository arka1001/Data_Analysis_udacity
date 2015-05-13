import numpy
import pandas


def normalize_features(array):
   """
   Normalize the features in our data set.
   """
   mu = array.mean()
   sigma = array.std()
   array_normalized = (array-mu)/sigma

   return array_normalized, mu, sigma


def compute_cost(features, values, theta):
    """
    Compute the cost of a list of parameters, theta, given a list of features 
    (input data points) and values (output data points).
    """
    m = len(values)
    sum_of_square_errors = numpy.square(numpy.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)

    return cost


def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    """
  
    m = len(values)
    cost_history = []
    cost = compute_cost(features, values, theta)
    cost_history.append(cost)

    for i in range(0,num_iterations):
        
        predicted_values = numpy.dot(features, theta)
        theta = theta + (1.0*alpha/m)*numpy.dot(values - predicted_values,features)
        cost = compute_cost(features, values, theta)
        cost_history.append(cost)

    return theta, pandas.Series(cost_history) 



if __name__ == "__main__":
    data = pandas.read_csv("../files/baseball_data.csv")
    features = data[['height', 'weight']]
    values = data[['HR']]
    m = len(values)

    features, mu, sigma = normalize_features(features)
    gradient_descent(features, values, theta,.01, 1000)


