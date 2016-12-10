import numpy as np
import sys
from scipy.spatial.distance import sqeuclidean
from scipy.cluster.vq import kmeans

k = 200
k_bicriterion = k / 10
epsilon = 1

def mapper(key, value):
	# key: None
	# value: one line of input file
	no_datapoints = value.shape[0]
	no_features = value.shape[1]

	# -----------------------
	# 1: Perform D^2 sampling
	# -----------------------
	# Assignment: data points -> centroid
	assignments = np.zeros(no_datapoints)
	distances = np.zeros(no_datapoints)

	# Sampling probabilites
	sampling_probablities = np.zeros(no_datapoints)
	sampling_probablities.fill(1. / no_datapoints)

	# Draw 1. centroid uniformly at random
	centroids = np.zeros((k_bicriterion, no_features))
	centroids[0, :] = value[np.random.choice(no_datapoints, 1, p=sampling_probablities), :].ravel()

	# Mapping that records which data points are assigned to the k'th centroid
	centroids_assignments = {k: [] for k in range(k_bicriterion)}

	# Draw "k_bicriterion - 1" subsequent centroids
	for t in range(k_bicriterion - 1):

		# Calculate the distance to the nearest centroid for 
		# each point and compute the sampling probs
		for i in range(no_datapoints):
		
			# Calculate distance to each selected centroid
			distance = sys.maxint
			assigned_centroid = -1
			for c in range(centroids.shape[0]):
				x = value[i, :].ravel()
				b = centroids[c, :].ravel()
				
				sqnorm = sqeuclidean(b, x)
				if (sqnorm < distance):
					assigned_centroid = c
					distance = sqnorm

			# Store the assigment and distance
			centroids_assignments[assigned_centroid].append(i)
			assignments[i] = assigned_centroid
			distances[i] = distance

		# The total distance
		sum_distances = np.sum(distances)

		# Calculate the sampling probabilites
		for i in range(no_datapoints):
			sampling_probablities[i] = float(distances[i]) / float(sum_distances)

		# Sample a new centroid
		centroids[t + 1] = value[np.random.choice(no_datapoints, 1, p=sampling_probablities), :].ravel()

	# The set "B": k bicriterion solutions (centroids)
	centroids = np.atleast_2d(centroids)

	# ------------------------------------------------
	# 2: Construct a coreset using importance sampling
	# ------------------------------------------------
	alpha = np.log(k) + 1
	phi = float(np.sum(distances)) / float(no_datapoints)

	# Calculate the sampling probabilies
	for i in range(no_datapoints):
		b = assignments[i]

		# For all data points assigned to the same centroid
		centroid_sum = np.sum([distances[j] for j in centroids_assignments[b]])

		term_1 = float(alpha * distances[i]) / phi
		term_2 = float(2.0 * alpha * centroid_sum) / (float(len(centroids_assignments[b])) * phi) 
		term_3 = float(4.0 * float(no_datapoints)) / float(len(centroids_assignments[b]))

		sampling_probablities[i] = term_1 + term_2 + term_3

	# Normalize
	total_sum = np.sum(sampling_probablities)
	sampling_probablities = [float(p) / float(total_sum) for p in sampling_probablities]
	
	# Construct a coreset
	coreset = value[np.random.choice(no_datapoints, 10*k, p=sampling_probablities), :]
	#coreset = value[np.random.choice(no_datapoints, int((no_features * (k_bicriterion**3)) / (epsilon**2)), p=sampling_probablities), :]

	yield "reduce", coreset


def reducer(key, values):
	# key: key from mapper used to aggregate
	# values: list of all value for that key
	# Note that we do *not* output a (key, value) pair here.
	X = np.atleast_2d(values)
	print(X.shape)

	# -------------------------------
	# 3: Run the k-means algorithm on
	# the merged coresets
	#--------------------------------
	centroids = kmeans(X, k)[0]
	yield centroids 
