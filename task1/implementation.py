import sys
import numpy as np
HASH_PRIME = 7817 # Choose some random prime number larger than the number of videos
RNG_SEED = 618984 # Random seed to be used by the RNG (so that each mapper derives the same hash functions)
NUM_VIDEOS = 700

R = 45
B = 21
def hashShingle(x, a, b):
    return ((a * x + b) % HASH_PRIME) % NUM_VIDEOS

def hashBand(xArr, coeffArr):
    hashSum = 0
    for i in range(0, R):
        hashSum += ((coeffArr[i, 0] * xArr[i] + coeffArr[i, 1]) % HASH_PRIME) % NUM_VIDEOS
    return hashSum % NUM_VIDEOS

def mapper(key, value):
    # key: None
    # value: one line of input file

    # Extract the videoID and shingles from the input line
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

    bandHashes = np.empty(B, dtype=int)
    for i in range(0, B):
        yield (i, hashBand(signatureVector[i*R:i*(R+1)], bandHashCoefficients[i*R:i*(R+1), :])), (videoId, shingles)

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    yield 1, 1

