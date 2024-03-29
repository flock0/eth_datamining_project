import numpy as np
from scipy.cluster.vq import kmeans, whiten, kmeans2
from math import log
import time


k = 200
d2SampleSize = 20
coresetSize = 2000
kMeansLoop = 20

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
    n = X.shape[0]
    lenB = B.shape[0]
    alpha = log(k) + 1
        
    # Calculate all squared distances beforehand
    distances = []
    for x in X:
        distances.append(squared_distance(x, B))
        
    c_mean = np.sum(distances) / (n * 1.0)
    
    nearest_B_indices = []
    nearbiDataPointsDistanceSums = np.zeros(lenB, dtype=float)
    nearbiDataPointsLength = np.zeros(lenB, dtype=int)
    
    for i, x in enumerate(X):
        iNearestB = nearest_B_index(x, B)
        nearest_B_indices.append(iNearestB)
        nearbiDataPointsDistanceSums[iNearestB] += distances[i]
        nearbiDataPointsLength[iNearestB] += 1.0

    sampling_probabilities = []
    for i, item in enumerate(X):
        first_term = alpha*distances[i]/c_mean
        
        second_term = (2.0*alpha)*nearbiDataPointsDistanceSums[nearest_B_indices[i]]/(nearbiDataPointsLength[nearest_B_indices[i]]*c_mean)
        
        third_term = 4.0 * n / nearbiDataPointsLength[nearest_B_indices[i]]

        sum_term = first_term + second_term + third_term
        sampling_probabilities.append(sum_term)

    
    sampling_probabilities = sampling_probabilities / (np.sum(sampling_probabilities) * 1.0)
    coreset_indices = np.random.choice(n,coresetSize,p=sampling_probabilities, replace=False)
    coreset = np.array(X[coreset_indices])
    weights = 1.0 / np.array(sampling_probabilities[coreset_indices])
    return coreset, weights

def mapper(key, value):
    print "Start mapper"
    print "============"
    # key: None
    # value: one line of input file
    print "Calculating d2sampling..."
    B = d2sampling(value, d2SampleSize)
    print "Calculating importancesampling..."
    coreset, weights = importance_sampling(value, B, coresetSize)
    print "finished"
    yield "key", np.concatenate([weights.reshape((coresetSize, 1)), coreset], axis=1)


def reducer(key, values):
    print "Start reducer"
    print "============="
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    splitArrays = np.split(values, [1],axis=1)
    weights = splitArrays[0].ravel()
    coreset = splitArrays[1]

    centroids,_ = kmeans(coreset, k, iter=kMeansLoop)

    # n = coreset.shape[0]
    # weights = weights / (1.0 * np.sum(weights))
    # centroids_indices = np.random.choice(coreset.shape[0], k, p=weights, replace=False)
    # centroids = np.array(coreset[centroids_indices])
    
    
    # print "Calculating gradient descent..." 
    # for l in range(0, kMeansLoop):
    #     print "Round", l
    #     for t in range(0, coreset.shape[0]):
    #         c = nearest_B_index(coreset[t], centroids)
    #         centroids[c] = centroids[c] + 10000.0 / (l*n+t+1) * weights[t] * (coreset[t] - centroids[c])
    
    print "Parameters"
    print "=========="

    print "k:",k
    print "d2SampleSize:",d2SampleSize
    print "coresetSize:",coresetSize
    print "kMeansLoop:",kMeansLoop

    yield centroids

#Oben 1000, unten 5000 coresetsize
