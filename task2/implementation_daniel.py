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

def caculateGradient(X, Y, lamda, w, batchsize, dimension):
    gradient = np.zeros(dimension,1)

    for i in range(0, batchsize):
        # Get the row
        x = X[:,i]
        y = Y[i]
        # Check if it is classified correctly with w
        # If not, add the negative direction to the gradient
        if (y*w.T.dot(x) < 1):
            gradient = gradient - np.multiply(y,x)

    # Add the content to the gradient
    gradient = gradient + np.multiply(lamda, w)
    return gradient



def mapper(key, value):
    # key: None
    # value: one line of input file
    # Implements the ADAM method

    # Hyper parameters
    lamda = 1
    alpha = 0.001
    beta0 = 0.9
    beta1 = 0.999
    epsilon = 1e-8
    T = 100
    batchsize = 128

    # Parse the input
    X,Y = parseValue(value)
    print(X.shape)
    print(Y.shape)

    # Get the dimension
    d = X.shape[1]

    # Initialize m_0, v_0 to zero
    m = np.zeros(d,1)
    v = np.zeros(d,1)

    # For t=1..T:
    for t in range(1,T):
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
    # Implements the EVA method (EVArage method, complementary to our ADAM)
    yield np.random.randn(400)
