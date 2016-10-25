import sys
import ast
import numpy as np
HASH_PRIME = 7817 # Choose some random prime number larger than the number of videos
RNG_SEED = 618984 # Random seed to be used by the RNG (so that each mapper derives the same hash functions)
NUM_VIDEOS = 700
SIMILARITY_THRESHOLD = 0.85
R = 13
B = 55


def hashShingle(x, a, b):
    """Calculate the hash for a shingle with parameter a and b

    Keyword arguments:
    x -- the shingle to hash
    a,b -- paramenter for our hash function
    """
    return ((a * x + b) % HASH_PRIME) % NUM_VIDEOS

def hashBand(xArr, coeffArr):
    """Calculate the hash for a band xArr with parameter coeffArr

    Keyword arguments:
    xArr -- the band to hash
    coeffArr -- Array of parameter for our hash function
    """
    hashSum = 0
    for i in range(0, R):
        hashSum += ((coeffArr[i, 0] * xArr[i] + coeffArr[i, 1]) % HASH_PRIME) % NUM_VIDEOS
    return hashSum % NUM_VIDEOS

def jaccardSimilarity(candidate1, candidate2):
    """Calculate the jaccardSimilarity of the two (videoId,videoShingle)-tuple candidate1 and candidate2

    Keyword arguments:
    candidate1, candidate2 -- two video tuple of (videoId, videoShingle) to calculate the jaccardSimilarity 
    """
    return np.intersect1d(candidate1[1], candidate2[1]).size / float(np.union1d(candidate1[1], candidate2[1]).size)

def mapper(key, value):
    """The Mapper for the mapReduce. It first calculate the signature vector and based on that the the hashbucket

    Keyword arguments:
    key -- None
    video -- one line of input

    Outputs:
    List of ((bandId, hash bucket of the hashed band), (videoId, shingles)) tuples
    """
    VIDEO_ID_STRING_LENGTH = 15
    VIDEO_ID_STRING_PREFIX_LENGTH = 6
    videoString = value[:VIDEO_ID_STRING_LENGTH]
    videoId = int(videoString[VIDEO_ID_STRING_PREFIX_LENGTH:])
    shingles = np.fromstring(value[VIDEO_ID_STRING_LENGTH + 1:], dtype=int, sep=' ')
    shingles.sort()

    # Decide on R * B functions for hashing the shingles
    np.random.seed(RNG_SEED)
    shingleHashCoefficients = np.random.randint(low=1, high=HASH_PRIME, size=(R*B, 2)) # Create random hash functions by drawing values for a and b
    # Decide on R functions for hashing a band
    bandHashCoefficients = np.random.randint(low=1, high=HASH_PRIME, size=(R, 2))

    # Create signature vector using min-hash
    signatureVector = np.full(R*B, sys.maxint)
    for shingle in shingles:
        for h in range(0, R*B):
            hashCoeffPair = shingleHashCoefficients[h]
            signatureVector[h] = min(signatureVector[h], hashShingle(shingle, hashCoeffPair[0], hashCoeffPair[1]))

     # Output candidate pairs in the following form:
     # key = (band, hash bucket of the hashed band)
     # value = (videoId, shingles)
     # For each band we output one pair.
     # The str() call is necessary as the key may not be a tuple, hence we convert it to a string.
    for i in range(0, B):
        yield str((i, hashBand(signatureVector[i*R:(i+1)*R], bandHashCoefficients))), (videoId, shingles)
    
def reducer(key, values):
    """The Reducer for the mapReduce. It checks all the pairs of values, if the Jaccard similarity is above the threshold

    Keyword arguments:
    key -- (bandId, hash bucket of the hashed band) tuples
    video -- (videoId, shingles)

    Outputs:
    a list of true pairs of candidate, which has a Jaccard similarity of at least the threshold
    """

    for c in range(0, len(values)):
        candidate1 = values[c]
        for candidate2 in values[c+1:]:
            if jaccardSimilarity(candidate1, candidate2) >= SIMILARITY_THRESHOLD:
                if candidate1[0] < candidate2[0]:
                    yield candidate1[0], candidate2[0]
                else:
                    yield candidate2[0], candidate1[0]

