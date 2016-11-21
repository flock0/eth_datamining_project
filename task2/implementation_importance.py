import numpy as np
import math
from sklearn.feature_selection import SelectKBest

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
    selection = SelectKBest()
    selection.fit(X,Y)

    yield "key",  selection.scores_ # This is how you yield a key, value pair


def reducer(key, values):
    '''
    key: key from mapper used to aggregate
    values: list of all value for that key
    Implements the EVA method (EVArage method, complementary to our ADAM)
    '''
    values = np.asmatrix(values)
    print(values.shape)
    np.set_printoptions(suppress=True)
    avg = np.average(values, axis=0)
    print avg
    yield np.zeros(400)