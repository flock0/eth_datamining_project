import numpy as np
from numpy.random import RandomState
import math
from time import sleep

def transform(X_input):
    """Calculate the random fourier features for an Gaussian kernel

    Keyword arguments:
    X_input - the input matrix X_input which will be transformed
    """

    # Parameters
    m = 3000
    gamma = 60

    # Copy
    X = X_input

    # Get the dimensions
    d = 0
    if len(X.shape)<=1:
        d = len(X)
    else:
        d = X.shape[1]

    # Draw iid m samples omega from p and b from [0,2pi]
    random_state = RandomState(124)
    omega = np.sqrt(2.0 * gamma) * random_state.normal(size=(d, m))
    b = random_state.uniform(0, 2 * np.pi, size=m)

    # Transform the input
    projection = np.dot(X, omega) + b
    Z = np.sqrt(2.0/m) * np.cos(projection)

    return Z

def parseValue(value):
    """ Parse the value from string to np matrix

    Keyword arguments:
    value - a string containing a matrix
    """
    arrayList = []
    for item in value:
        array = np.fromstring(item, dtype=float, sep=' ')
        arrayList.append(array)
    matrix = np.asarray(arrayList)
    X = matrix[:,1:]
    X_trans = transform(X)
    Y = matrix[:,0]
    return X_trans,Y

def calculateGradient(X, Y, lamda, w, batchsize, dimension):
    """ Calculate the gradient of a hinge loss

    Keyword arguments:
    value - a string containing a matrix
    """

    # random batch selection
    index = np.random.choice(X.shape[0], size = batchsize, replace=False)
    
    # Gradient calculation
    gradient = np.zeros(dimension)
    counter = 0

    for i in index:
        # Get the row
        x = np.ravel(X[i,:])
        y = Y[i]
        # Check if it is classified correctly with w
        # If not, add the negative direction to the gradient
        if y*w.dot(x) < 1:
            gradient = gradient - y*x
            counter = counter + 1
    print("Counter",counter)

    return gradient



def mapper(key, value):
    '''
    The mapper which implements the PEGASOS method (complementary to our EVA method)
    key: None
    value: one line of input file
    '''

    # Hyper parameters
    lamda = 1.7e-5 # BEST: 1.7 with 0.8169
    T = 2000

    batchsize = 1024 # unten links 1500
    alpha = 0.5

    # Parse the input and shuffle it
    X,Y = parseValue(value)
    # TODO: shuffle the input
 
    # Get the dimension and samplesize
    d = X.shape[1]
    n = X.shape[0]

    # Initialize m_0, v_0, w_0 to zero
    w = np.zeros(d)
    gradient_old = np.zeros(d)

    for t in range(1,T):
        # Calculate the gradient g_t
        gradient = calculateGradient(X,Y,lamda,w,batchsize,d)
        gradient = alpha*gradient_old + (1-alpha) * gradient
        # Update the weight vector
        w = (1-(1/float(t))) * w - ((1/(lamda*float(t)))/batchsize) * gradient
        w = min(1, ((1/np.sqrt(lamda))/np.linalg.norm(w))) * w

        gradient_old = gradient
    yield "key", w  # This is how you yield a key, value pair


def reducer(key, values):
    '''
    Implements the EVA method (EVArage method, complementary to our ADAM)
    key: key from mapper used to aggregate
    values: list of all value for that key
    '''
    avg = np.average(values, axis=0)
    yield avg