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

def calculateGradient(X, Y, lamda, w, batchsize, dimension):
    index = np.random.choice(X.shape[0], size = batchsize, replace=False)
    gradient = np.zeros(dimension)
    counter = 0

    for i in index:
        # Get the row
        x = X[i,:]
        y = Y[i]
        # Check if it is classified correctly with w
        # If not, add the negative direction to the gradient
        #print("check", y*w.dot(x))
        if y*w.dot(x) < 1:
            gradient = gradient - y*x
            counter = counter + 1
    print("Counter",counter)

    return gradient



def mapper(key, value):
    '''key: None
    value: one line of input file
    Implements the ADAM method (complementary to our EVA method)
    '''

    # Hyper parameters
    lamda = 10e-13
    T = 100
    batchsize = 1024

    # Parse the input and shuffle it
    X,Y = parseValue(value)
    # TODO: shuffle the input

    # Get the dimension and samplesize
    d = X.shape[1]
    n = X.shape[0]

    # Initialize m_0, v_0, w_0 to zero
    m = np.zeros(d)
    v = np.zeros(d)
    w = np.zeros(d)

    for t in range(1,T):
        # Calculate the gradient g_t
        gradient = calculateGradient(X,Y,lamda,w,batchsize,d)
        # Update the weight vector
        w = (1-(1/t)) * w - ((1/(lamda*t))/batchsize) * gradient
        w = min(1, ((1/np.sqrt(lamda))/np.linalg.norm(w))) * w
        #sprint("new weight vector:", w)
    yield "key", w  # This is how you yield a key, value pair


def reducer(key, values):
    '''
    key: key from mapper used to aggregate
    values: list of all value for that key
    Implements the EVA method (EVArage method, complementary to our ADAM)
    '''
    avg = np.average(values, axis=0)
    print(avg)
    yield avg