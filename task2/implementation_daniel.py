import numpy as np
import math

def transform(X_input):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    X = X_input
    print(X.shape)
    for i in range(X.shape[1]):
        new_feature = np.multiply(X[:,i],X[:,i]).reshape(X.shape[0],1)
        np.append(X, new_feature, axis=1)
        new_feature2 = np.multiply(np.multiply(X[:,i],X[:,i]),X[:,i]).reshape(X.shape[0],1)
        np.append(X, new_feature2, axis=1)
        new_feature3 = np.sqrt(X[:,i]).reshape(X.shape[0],1)
        np.append(X, new_feature3, axis=1)
        new_feature4 = np.exp(-X[:,i]).reshape(X.shape[0],1)
        np.append(X, new_feature4, axis=1)
        new_feature5 = np.exp(X[:,i]).reshape(X.shape[0],1)
        np.append(X, new_feature5, axis=1)
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

def calculateGradient(X, Y, lamda, w, batchsize, dimension):
    index = np.random.choice(X.shape[0], size = batchsize, replace=False)
    gradient = np.zeros(dimension)
    counter = 0

    for i in index:
        # Get the row
        x = np.ravel(X[i,:])
        y = Y[i]
        # Check if it is classified correctly with w
        # If not, add the negative direction to the gradient
        #print("check", y*w.dot(x))
        if y*w.dot(x) < 1:
            gradient = gradient - y*x
            counter = counter + 1
    print("Counter",counter)

    # Add the constent to the gradient
    gradient = gradient + lamda*w
    return gradient



def mapper(key, value):
    '''key: None
    value: one line of input file
    Implements the ADAM method (complementary to our EVA method)
    '''

    # Hyper parameters
    lamda = 1e-7
    alpha = 1e-3
    beta1 = 0.99
    beta2 = 0.9
    epsilon = 1e-8
    T = 1000
    batchsize = 512

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
        #print("gradient", gradient)
        # Calculate the first moment m_t
        m = beta1 * m + (1-beta1) * gradient
        #print("first moment:", m)
        # Calculate the second moment v_t
        v = beta2 * v + (1-beta2) * (gradient ** 2)
        #print("second moment:", v)
        # Calculate the first bias correction
        m = m / (1 - (beta1 ** t))
        #print("first correction", m)
        # Calculate the second bias correction
        v = v / (1 - (beta2 ** t))
        #print("second correction:", m)
        # Update the weight vector
        w = w - alpha * m / (np.sqrt(v) + epsilon)
        #sprint("new weight vector:", w)
        # Constraint
        w = min(1, ((1/np.sqrt(lamda))/np.linalg.norm(w))) * w
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