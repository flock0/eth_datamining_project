import math
import numpy as np
from time import sleep


X = {}
B = {}
B_T = {}
b = {}
phi_hat = {}
A = {}
A_inv = {}
Ainv_x_products = {}

d = 6 # article features
k = 36 # user features * article features
delta = 0.05
alpha = 1 + math.sqrt(math.log(2.0 / delta) / 2)

best_article = -1
best_x = -1
best_z = -1

@profile
def inverse(A):
    return np.linalg.inv(A)

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
    beta_hat = A_0_inv.dot(b[0]).ravel()

@profile
def update(reward):
    '''
    Update the weights
    '''

    # Check if the reward is positive
    if reward == -1:
        return


    # Set all the variables to global
    global A_0
    global A_0_inv
    global beta_hat
    global A
    global A_inv
    global Ainv_x_products
    global B
    global B_T
    global b
    global phi_hat
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
    Ainv_x_products[best_article] = np.inner(A_inv[best_article], x_T) # We cache this product so we don't have to recalculate it for every recommend step
    B[best_article] += np.outer(x_T, z_T)
    B_T[best_article] = np.transpose(B[best_article])

    b[best_article] += reward * x_T.reshape((d,1))

    BT_Ainv_product = np.transpose(B[best_article]).dot(A_inv[best_article]) # Dot product used for the next two subtractions
    A_0 += np.outer(z_T, z_T) - BT_Ainv_product.dot(B[best_article])
    b[0] += reward * z_T.reshape((k,1)) - BT_Ainv_product.dot(b[best_article])

    # Update A_0_inv an beta_hat here, as it does not get modified in the recommend function at all
    A_0_inv = inverse(A_0)
    beta_hat = A_0_inv.dot(b[0]).ravel()
    for article in A:
        phi_hat[article] = A_inv[article].dot(b[article].ravel() - B[article].dot(beta_hat))

step = 0
@profile
def recommend(time, user_features, choices):

    # Set all the variables to global
    global A
    global A_inv
    global Ainv_x_products
    global B
    global B_T
    global b
    global phi_hat
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
        x_T = X[article]
        if not article in A:
            A[article] = np.identity(d)
            A_inv[article] = np.identity(d)
            Ainv_x_products[article] = np.inner(A_inv[article], x_T)
            B[article] = np.zeros([d,k])
            B_T[article] = np.zeros([k,d])
            b[article] = np.zeros([d,1])
            phi_hat[article] = A_inv[article].dot(b[article].ravel() - B[article].dot(beta_hat))
        # Get the row vector of the current article
        

        z_T = np.outer(x_T,user_features_array).ravel()
        
        B_a = B[article]
        B_a_T = B_T[article]

        # Estimate our phi_a_hat
        A_a_inv = A_inv[article]
        # print "A_a_inv:", A_a_inv.shape
        # print "b[article]:", b[article].shape
        # print "B[article]:", B[article].shape
        # print "beta_hat:", beta_hat.shape

        phi_a_hat = phi_hat[article]

        # Calculate the variance s_t_a
        # print "z_T:", z_T.shape
        # print "A_0_inv:", A_0_inv.shape
        # print "z:", z.shape
        first_term = np.inner(z_T.dot(A_0_inv), z_T)
        # print "first_term:", first_term

        # print "z_T", z_T.shape
        # print "A_0_inv:", A_0_inv.shape 
        # print "B_a_T:", B_a_T.shape
        # print "A_a_inv:", A_a_inv.shape
        # print "x:", x.shape 
        Ainv_x_product = Ainv_x_products[article] #blau
        A0inv_BT_Ainv_x_product = np.inner(A_0_inv, np.inner(B_a_T, Ainv_x_product)) #grean
        second_term = 2 * np.inner(z_T, A0inv_BT_Ainv_x_product)
        # print "second_term:", second_term 

        third_term = np.inner(x_T, Ainv_x_product)
        # print "third_term:", third_term
        
        forth_term = np.inner(x_T, np.inner(A_a_inv, np.inner(B_a, A0inv_BT_Ainv_x_product)))
        # print "forth_term:", forth_term

        s_t_a = first_term - second_term + third_term + forth_term

        # print "z_T:", z_T.shape
        # print "beta_hat:", beta_hat.shape
        # print "x:", x.shape
        # print "phi_a_hat:", phi_a_hat.shape
        pta_first_term = np.inner(z_T, beta_hat)
        pta_second_term = np.inner(x_T, phi_a_hat)
        p_t_a = pta_first_term + pta_second_term + alpha * math.sqrt(s_t_a)
        # print "p_t_a:", p_t_a

        if p_t_a > best_score:
            best_article = article
            best_score = p_t_a
            best_x = x_T
            best_z = z_T


    return best_article














