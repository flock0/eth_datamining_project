# Task1: Duplicate-detection using Locality Sensitive Hashing

Mapper: After splitting the video ID from the shingles we initialize the hash functions by drawing the “a” and “b” coefficients of the hash functions. To make sure that each mapper comes up with the same hash functions, we statically set the seed for the RNG. Besides defining the coefficients for the shingle hash functions, we also set up the hash functions for band-wise hashing.
As each mapper processes one video it is responsible for a single vector of the signature matrix. The input already denotes which shingles occur in the video, hence we can simply loop through all the available shingles, hash them with our previously created hash functions and keep the minimum in the signature vector.
As a final step the mapper loops through the available bands and hashes them as well.
The mapper outputs a key-value pair, with the key being a tuple with the band and the hash bucket. The value contains the video ID and the shingles.

Through the shuffle phase the candidate pairs get grouped together, whereas videos whose bands got hashed to the same (band-specific) bucket are processed together. All the reducers need to do is do a pair-wise calculation of the jaccard similarity and output the video IDs, if the similarity is higher or equal to the defined threshold.

We chose R= 13 and B = 55. With those values we want to make sure that candidate pairs with a similarity of exactly 0.85 are to be considered in any case, and that no false negatives can occur.
