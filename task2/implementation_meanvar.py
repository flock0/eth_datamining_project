import numpy as np
import math

def transform(X):
    return X

def parseValue(value):
    arrayList = []
    for item in value:
        array = np.fromstring(item, dtype=float, sep=' ')
        arrayList.append(array)
    matrix = np.asarray(arrayList)
    X = matrix[:,1:]
    X_trans = transform(X)
    Y = matrix[:,0]
    return X_trans,Y

def mapper(key, value):
    '''key: None
    value: one line of input file
    Implements the ADAM method (complementary to our EVA method)
    '''

    # Parse the input and shuffle it
    X,Y = parseValue(value)
    
    meanList = []
    stdList = []

    for i in range(X.shape[1]):
        feature = X[:,i]
        meanList.append(np.average(feature))
        stdList.append(np.std(feature))

    meanList = np.asarray(meanList)
    stdList = np.asarray(stdList)

    yield "key", (meanList, stdList)  # This is how you yield a key, value pair


def reducer(key, values):
    '''
    key: key from mapper used to aggregate
    values: list of all value for that key
    Implements the EVA method (EVArage method, complementary to our ADAM)
    '''
    meanCollection = []
    stdCollection = []

    for (meanList, stdList) in values:
        meanCollection.append(meanList)
        stdCollection.append(stdList)

    meanCollection = np.asarray(meanCollection)
    stdCollection = np.asarray(stdCollection)
    
    # TODO: Calculate the mean of all means and standard deviation
    print meanCollection.shape
    print stdCollection.shape

    mean = np.average(meanCollection, axis=0)
    std = np.average(stdCollection, axis=0)

    print mean
    print std
    
    yield np.zeros(400)