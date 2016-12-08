import numpy as np
from scipy.cluster.vq import kmeans

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
	Iteratively sample data points as new centers proportianal
	to squared distance to existing cluster centers

	Arguments:
	X - The data points as a matrix
	samplesize - the number of sample which should be drawn
	'''
	# Copy the input
	n = X.shape[0]
	d = X.shape[1]

	# Initialize the list of drawn samples
	index_list = []
	B = []

	# Draw the first sample uniformly at random
	index = np.random.choice(n)
	index_list.append(index)
	B.append(X[index,:])

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
		index_list.append(index)
		sample = X[index,:].ravel()
		B.append(sample)

	# return the result as numpy array
	B = np.asmatrix(B)
	return B

def importance_sampling():
	print "Not implemented yet"

def mapper(key, value):
    # key: None
    # value: one line of input file
    B = d2sampling(value, 5)
    yield "key", "value"  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    print len(values)
    yield np.random.randn(200, 250)

