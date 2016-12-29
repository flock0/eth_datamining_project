import math
import numpy as np
from time import sleep
from numpy.linalg import lapack_lite
lapack_routine = lapack_lite.dgesv

X = {}
B = {}
b = {}
A = {}
A_inv = {}

d = 6 # article features
k = 36 # user features * article features
delta = 0.05
alpha = 1 + math.sqrt(math.log(2.0 / delta) / 2)

best_article = -1
best_x = -1
best_z = -1

# https://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy


@profile
def inverse(A):

    n_eq = A.shape[0]
    n_rhs = A.shape[1]
    identity  = np.eye(n_eq)
    b = np.copy(identity)
    pivots = np.zeros(n_eq, np.intc)
    results = lapack_lite.dgesv(n_eq, n_rhs, A, n_eq, pivots, b, n_eq, 0)
    if results['info'] > 0:
        raise LinAlgError('Singular matrix')
    return b

@profile
def set_articles(articles):
    '''
    Save the articles globally as a hashmap and initialize for hybrid LinUCB

    Arguments:
    articles - a hashmap of all articles
    '''

    global X
    # Store all the article features as row vectors in the X dictionary
    for key, value in articles.iteritems():
        X[key] = np.array(value)

    # Initialize A_0 and b_0
    global A_0
    global A_0_inv
    global beta_hat
    global A
    global b

    # Initialize A_0_inv an beta_hat here as well, as it does not get modified in the recommend function at all
    A_0 = np.identity(k)
    A_0_inv = np.identity(k)
    b[0] = np.zeros([k,1])
    beta_hat = A_0_inv.dot(b[0])

@profile
def update(reward):
    '''
    Update the weights
    '''
    # print ""
    # print "update"
    # print "======"

    # Check if the reward is positive
    if reward == -1:
        return

    # Set all the variables to global
    global A_0
    global A_0_inv
    global beta_hat
    global A
    global A_inv
    global B
    global b
    global X
    global best_article
    global best_score
    global best_x
    global best_z

    # Update the weights
    B_a = B[best_article]
    B_a_T = np.transpose(B_a)
    A_a = A[best_article]
    A_a_inv = A_inv[best_article]
    x_T = best_x
    z_T = best_z
    
    BT_Ainv_product = B_a_T.dot(A_a_inv) # Dot product used for the next two sums
    A_0 += BT_Ainv_product.dot(B_a)
    b[0] += BT_Ainv_product.dot(b[best_article])

    A[best_article] += np.outer(x_T, x_T)
    A_inv[best_article] = inverse(A[best_article]) # We cache A_inv so we don't have to recalculate it for every recommend step
    B[best_article] += np.outer(x_T, z_T)

    b[best_article] += reward * x_T.reshape((d,1))

    BT_Ainv_product = np.transpose(B[best_article]).dot(A_inv[best_article]) # Dot product used for the next two subtractions
    A_0 += np.outer(z_T, z_T) - BT_Ainv_product.dot(B[best_article])
    b[0] += reward * z_T.reshape((k,1)) - BT_Ainv_product.dot(b[best_article])

    # Update A_0_inv an beta_hat here, as it does not get modified in the recommend function at all
    A_0_inv = inverse(A_0)
    beta_hat = A_0_inv.dot(b[0])

step = 0
@profile
def recommend(time, user_features, choices):

    # Set all the variables to global
    global A
    global A_inv
    global B
    global b
    global X
    global best_article
    global best_score
    global best_x
    global best_z
    
    user_features_array = np.array(user_features)
    # Estimate the score per article
    best_article = -1
    best_score = -1
    for article in choices:
        if not article in A:
            A[article] = np.identity(d)
            A_inv[article] = np.identity(d)
            B[article] = np.zeros([d,k])
            b[article] = np.zeros([d,1])
            
        # Get the row vector of the current article
        x_T = X[article]

        z_T = np.outer(x_T,user_features_array).ravel()
        
        B_a = B[article]
        B_a_T = np.transpose(B[article])

        # Estimate our phi_a_hat
        A_a_inv = A_inv[article]
        # print "A_a_inv:", A_a_inv.shape
        # print "b[article]:", b[article].shape
        # print "B[article]:", B[article].shape
        # print "beta_hat:", beta_hat.shape

        phi_a_hat = A_a_inv.dot(b[article] - B[article].dot(beta_hat))

        # Calculate the variance s_t_a
        # print "z_T:", z_T.shape
        # print "A_0_inv:", A_0_inv.shape
        # print "z:", z.shape
        zT_A0inv_prodcuct = z_T.dot(A_0_inv)  # Dot product used for the next two terms
        first_term = np.inner(zT_A0inv_prodcuct, z_T)
        # print "first_term:", first_term

        # print "z_T", z_T.shape
        # print "A_0_inv:", A_0_inv.shape 
        # print "B_a_T:", B_a_T.shape
        # print "A_a_inv:", A_a_inv.shape
        # print "x:", x.shape 
        second_term = np.inner((2 * zT_A0inv_prodcuct).dot(B_a_T).dot(A_a_inv), x_T)
        # print "second_term:", second_term 

        xT_Aainv_product = np.inner(x_T, A_a_inv) # Inner product used for the next two terms
        third_term = np.inner(xT_Aainv_product, x_T)
        # print "third_term:", third_term
        
        forth_term = np.inner(xT_Aainv_product.dot(B_a).dot(A_0_inv).dot(B_a_T).dot(A_a_inv), x_T)
        # print "forth_term:", forth_term

        s_t_a = first_term - second_term + third_term + forth_term

        # print "z_T:", z_T.shape
        # print "beta_hat:", beta_hat.shape
        # print "x:", x.shape
        # print "phi_a_hat:", phi_a_hat.shape
        
        p_t_a = z_T.dot(beta_hat)[0] + np.inner(x_T, phi_a_hat.ravel()) + alpha * np.sqrt(s_t_a * 1.0)
        # print "p_t_a:", p_t_a

        if p_t_a > best_score:
            best_article = article
            best_score = p_t_a
            best_x = x_T.reshape((d,1))
            best_z = z_T


    return best_article














