import numpy as np
HASH_PRIME = 7817 # Choose some random prime number larger than the number of videos
RNG_SEED = 618984 # Random seed to be used by the RNG (so that each mapper derives the same hash functions)
NUM_VIDEOS = 700

R = 45
B = 21
def hash(x, a, b):
    return ((a * x + b) % HASH_PRIME) % NUM_VIDEOS
def mapper(key, value):
    # key: None
    # value: one line of input file

    # Extract the videoID and shingles from the input line
    VIDEO_ID_STRING_LENGTH = 15
    VIDEO_ID_STRING_PREFIX_LENGTH = 6
    videoString = value[:VIDEO_ID_STRING_LENGTH]
    videoId = int(videoString[VIDEO_ID_STRING_PREFIX_LENGTH:])
    shingles = np.fromstring(value[VIDEO_ID_STRING_LENGTH + 1:], dtype=int, sep=' ')
    
    # Decide on R * B hash functions
    np.random.seed(RNG_SEED)
    hash_coefficients = np.random.randint(low=1, high=HASH_PRIME, size=(R * B, 2)) # Create random hash functions by drawing values for a and b

    yield 1, 1

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    yield 1, 1

