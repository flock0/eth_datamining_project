import numpy as np
from scipy.cluster.vq import kmeans, whiten, kmeans2
from math import log
import time


k = 200
d2SampleSize = 20
coresetSize = 900
kMeansLoop = 16

def nearest_B_index(x, B):
    '''
    Returns the index of the nearest point in B to point x
    
    Arguments:
    x - a data point
    B - a set of data points
    '''
    min_distance = float("inf")
    nearest_B_index = -1
    
    for i in range(0, B.shape[0]):
        dist = np.sum(np.power((x - B[i, :].ravel()),2))
        if dist < min_distance:
            min_distance = dist
            nearest_B_index = i

    if nearest_B_index == -1:
        print "Error in nearest_B_index"
    return nearest_B_index

def squared_distance(x, B):
    '''
    Calculates the minimum distance of a point and a set of points
    
    Arguments:
    x - a data point
    B - a set of data points
    '''
    min_distance = float("inf")
    for item in B:
        dist = np.sum(np.power((x - item),2))
        min_distance = min(min_distance, dist)
    return min_distance

def d2sampling(X, samplesize):
    '''
    Iteratively sample data points as new centers proportional
    to squared distance to existing cluster centers

    Arguments:
    X - The data points as a matrix
    samplesize - the number of sample which should be drawn
    '''
    # Copy the input
    n = X.shape[0]
    d = X.shape[1]

    # Initialize the list of drawn samples
    B = []

    # Draw the first sample uniformly at random
    index = np.random.choice(n)
    B.append(X[index,:].ravel())

    # Draw the rest of the samples according to their distance to existing samples
    for i in range(samplesize-1):
        # Calculate the probability for each sample data
        weights = []
        for x in X:
            sqrt_dist = squared_distance(x, B)
            weights.append(sqrt_dist)
            
        # Normalize the weights
        weights = weights / (np.sum(weights) * 1.0)

        # Draw the next sample
        index = np.random.choice(n,1,p=weights)
        sample = X[index,:].ravel()
        B.append(sample)

    # return the result as numpy array
    B = np.asmatrix(B)
    return B

def importance_sampling(X, B, coresetSize):
    '''
    Samples points with high impact on the objective function,
    which upper bound the sensitivity function, more frequently.

    Arguments:
    X - The data points as a matrix
    B - The set of bicriteria points as a matrix
    coresetSize - the number of sample which should be drawn
    '''
    n = X.shape[0]
    lenB = B.shape[0]
    alpha = log(k, 2) + 1
        
    # Calculate all squared distances beforehand
    distances = []
    for x in X:
        distances.append(squared_distance(x, B))
    
    # Calculate the mean of square distances
    c_mean = np.sum(distances) / (n * 1.0)
    
    # Initialize the arrays for the importance sampling calculation
    nearest_B_indices = []
    nearbiDataPointsDistanceSums = np.zeros(lenB, dtype=float)
    nearbiDataPointsLength = np.zeros(lenB, dtype=float)
    
    # Prepare the arrays for the importance sampling calculation
    for i, x in enumerate(X):
        iNearestB = nearest_B_index(x, B)
        nearest_B_indices.append(iNearestB)
        nearbiDataPointsDistanceSums[iNearestB] += distances[i]
        nearbiDataPointsLength[iNearestB] += 1.0

    # Calculate the importance sampling distribution
    sampling_probabilities = []
    for i, item in enumerate(X):
        first_term = alpha*distances[i]/c_mean
        
        second_term = (2.0*alpha)*nearbiDataPointsDistanceSums[nearest_B_indices[i]]/(nearbiDataPointsLength[nearest_B_indices[i]]*c_mean)
        
        third_term = 4.0 * n / nearbiDataPointsLength[nearest_B_indices[i]]

        sum_term = first_term + second_term + third_term
        sampling_probabilities.append(sum_term)
    
    # Normalize the sampling distribution
    sampling_probabilities = sampling_probabilities / (np.sum(sampling_probabilities) * 1.0)
    
    # Draw the coreset in respect to the sampling distribution
    coreset_indices = np.random.choice(n,coresetSize,p=sampling_probabilities, replace=False)
    coreset = np.array(X[coreset_indices])
    
    # Calculate the weight
    weights = 1.0 / np.array(sampling_probabilities[coreset_indices])
    return coreset, weights

def mapper(key, value):
    '''
    sample points with high impact on the objective function,
    which upper bound the sensitivity function, more frequently.

    Arguments:
    key - None
    value - the data set
    '''
    # Sample with d2 sampling to get the bicriteria set B
    B = d2sampling(value, d2SampleSize)
    # Sample the coresetk with importance sampling
    coreset, weights = importance_sampling(value, B, coresetSize)
    # return the coreset with it's weights
    yield "key", np.concatenate([weights.reshape((coresetSize, 1)), coreset], axis=1)

def reducer(key, values):
    '''
    Caculates the centroids from the coresets by using kmeans with
    Llyod's heuristic. The points are initialized weighted by the
    inverse of their weights.

    Arguments:
    key - fixed as "key"
    value - the coreset with it0s weights continated
    '''
    # Split the weights and the coreset
    splitArrays = np.split(values, [1],axis=1)
    weights = splitArrays[0].ravel()
    coreset = splitArrays[1]

    # Get the parameters
    n = coreset.shape[0]
    d = coreset.shape[1]

    # Update the weights
    prob = 1.0 / weights
    prob = prob / (np.sum(prob) * 1.0)
    weights = weights / (1.0 * np.sum(weights))

    # Initialize k centroids
    centroids_indices = np.random.choice(coreset.shape[0], k, p=prob, replace=False)
    centroids = np.array(coreset[centroids_indices])
    old_centroids = np.zeros((k,d))
    
    # Kmeans algorithm with Lloyd's heuristic
    round = 0
    while True:
        old_centroids = np.copy(centroids)

        # Initialize the centroid mapper
        centroids_sum = np.zeros((k,d))
        centroids_weight = np.zeros(k)

        # For each point, do the assignment to the nearest centroid
        for i, x in enumerate(coreset):
            index_nearest_B = nearest_B_index(x, centroids)
            centroids_sum[index_nearest_B] += x * weights[i]
            centroids_weight[index_nearest_B] += weights[i]

        # update the centroid center
        for i in range(k):
            if centroids_weight[i] > 1e-7:
                centroids[i] = centroids_sum[i] / centroids_weight[i]
            else:
                print "centroid", i, "has 0 weight"

        # Check convergence
        if np.array_equal(old_centroids,centroids):
            break

        # Check Max round
        if round > kMeansLoop:
            break
            
    yield centroids
