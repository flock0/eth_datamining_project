import numpy as np

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    return X

def parseValue(value):
    arrayList = []
    for item in value:
        array = np.fromstring(item, dtype=float, sep=' ')
        arrayList.append(array)
    matrix = np.asarray(arrayList)
    X = matrix[:,1:]
    Y = matrix[:,1]
    return X,Y

def caculateGradient(X, Y, lamda, w, batchsize):
    random_rows = np.random.choice(X.shape[0], size=batchsize, replace=False)
    
    gradient = 0

    for row in random_rows:
        x_row = X[:,row]
        # Check if it is classified correctly with w
        # If not, add the negative direction to the gradient

    # Add the content to the gradient



def mapper(key, value):
    # key: None
    # value: one line of input file
    X,Y = parseValue(value)
    print(X.shape)
    print(Y.shape)

    # Initialize m_0, v_0 to zero

    # For t=1..T:
        # Calculate the gradient g_t
        # Calculate the first moment m_t
        # Calculate the second moment v_t
        # Calculate the first bias correction
        # Calculate the second bias correction
        # Update the weight vector

    
    yield "key", "value"  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield np.random.randn(400)
