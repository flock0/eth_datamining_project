import numpy as np
import itertools

shingle_range = [0, 8192]

#parameters for hashing
prime = 56267
N = 8192
H = 1000
B = 1
b = H/B
hash_family = np.random.randint(1,10000,(H,2))

bucket_hash = np.random.randint(1,10000,(B,2))

def hash(col, a, b):
    return ((a*float(col) + b) % prime) % N


def mapper(key, value):
    # key: None
    # value: one line of input file
    input = value.split()
    title = input[0]
    values = input[1:]

    #initialize signature matrix
    M = np.full((H,1), float('inf'))

    for entry in values:
        for i in range(0,H-1):
            h = hash_family[i]
            temp_hash = hash(entry, h[0], h[1])
            if M[i] > temp_hash:
                M[i] = temp_hash

    buckets = np.full((B,1), float('inf'))
    for i in range(0, B):
        for j in range(0,b):
            input_a = bucket_hash[i,0]
            input_b = bucket_hash[i,1]
            temp_hash = hash(M[i*b+j], input_a, input_b)
            if buckets[i] > temp_hash:
                buckets[i] = temp_hash

    sum = (np.sum(buckets) % N)
    yield (sum, value)

def jaccard_sim(tuple):
    intersection = tuple[0].intersection(tuple[1])
    union = tuple[0].union(tuple[1])
    return float(len(intersection))/len(union)

def reducer(key, values):
   # key: key from mapper used to aggregate
    # values: list of all value for that key

    set_list = list()
    for vid in values:
        split_vid = vid.split()
        vid_id = int(split_vid[0].split('_')[1])
        set_list.append((vid_id, set(split_vid[1:])))

    for pair in itertools.combinations(set_list, 2):
        sim = jaccard_sim((pair[0][1], pair[1][1]))
        if sim >= 0.85:
            if pair[0][0] <= pair[1][0]:
                yield (pair[0][0], pair[1][0])
            else:
                yield  (pair[1][0], pair[0][0])