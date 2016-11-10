import numpy as np
import itertools

shingle_range = [0, 8192]

#parameters for hashing
prime = 94261
N = 700
R = 10
B = 40
H = R*B


hash_family = np.random.randint(1,prime,(H,2))
bucket_hash_family = np.random.randint(1,prime,(R,2))

#hash function that hashes column col with hash coefficients a and b
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
        for i in range(0,H):
            h = hash_family[i]
            temp_hash = hash(entry, h[0], h[1])
            if M[i] > temp_hash:
                M[i] = temp_hash

    for i in range(0, B):
    	bucket_signature = []
        for j in range(0,R):
            input_a = bucket_hash_family[j,0]
            input_b = bucket_hash_family[j,1]
            bucket_signature.append(hash(M[i*R+j], input_a, input_b))
        bucket_hash = np.sum(bucket_signature) % N
        yield str((i, bucket_hash)), value


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
