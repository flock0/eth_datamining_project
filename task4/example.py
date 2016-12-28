import math
import numpy as np
from time import sleep


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

def set_articles(articles):
    '''
    Save the articles globally as a hashmap and initialize for hybrid LinUCB

    Arguments:
    articles - a hashmap of all articles
    '''
    global X
    X = articles

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

def update(reward):
    '''
    Update the weights
    '''

    # Check if the reward is positive
    if reward == -1:
        pass


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
    x = best_x
    z = best_z
    
    BT_Ainv_product = B_a_T.dot(A_a_inv) # Dot product used for the next two sums
    A_0 += BT_Ainv_product.dot(B_a)
    b[0] += BT_Ainv_product.dot(b[best_article])

    A[best_article] += np.outer(x.ravel(),x.ravel())
    A_inv[best_article] = np.linalg.inv(A[best_article]) # We cache A_inv so we don't have to recalculate it for every recommend step
    B[best_article] += np.outer(x.ravel(),z.ravel())

    b[best_article] += reward * x

    BT_Ainv_B_product = np.transpose(B[best_article]).dot(A_inv[best_article]) # Dot product used for the next two subtractions
    A_0 += np.outer(z.ravel(),z.ravel()) - BT_Ainv_B_product.dot(B[best_article])
    b[0] += reward * z - BT_Ainv_B_product.dot(b[best_article])

    # Update A_0_inv an beta_hat here, as it does not get modified in the recommend function at all
    A_0_inv = np.linalg.inv(A_0)
    beta_hat = A_0_inv.dot(b[0])

step = 0

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
    
    # Estimate the score per article
    best_article = -1
    best_score = -1
    for article in choices:
        if not article in A:
            A[article] = np.identity(d)
            A_inv[article] = np.identity(d)
            B[article] = np.zeros([d,k])
            b[article] = np.zeros([d,1])
            
        # Calculate x
        x = np.array(X[article]).reshape((d,1))
        x_T = np.transpose(x)

        z = np.outer(x.ravel(),np.array(user_features).ravel()).reshape(k,1)
        z_T = np.transpose(z)

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
        first_term = z_T.dot(A_0_inv).dot(z)[0][0]
        # print "first_term:", first_term

        # print "z_T", z_T.shape
        # print "A_0_inv:", A_0_inv.shape 
        # print "B_a_T:", B_a_T.shape
        # print "A_a_inv:", A_a_inv.shape
        # print "x:", x.shape 
        second_term = 2 * z_T.dot(A_0_inv).dot(B_a_T).dot(A_a_inv).dot(x)[0]
        # print "second_term:", second_term 

        third_term = x_T.dot(A_a_inv).dot(x)
        # print "third_term:", third_term
        
        forth_term = x_T.dot(A_a_inv).dot(B_a).dot(A_0_inv).dot(B_a_T).dot(A_a_inv).dot(x)
        # print "forth_term:", forth_term

        s_t_a = first_term - second_term + third_term + forth_term

        # print "z_T:", z_T.shape
        # print "beta_hat:", beta_hat.shape
        # print "x:", x.shape
        # print "phi_a_hat:", phi_a_hat.shape
        p_t_a = z_T.dot(beta_hat)[0][0] + x_T.dot(phi_a_hat)[0][0] + alpha * np.sqrt(s_t_a * 1.0)
        # print "p_t_a:", p_t_a

        if p_t_a > best_score:
            best_article = article
            best_score = p_t_a
            best_x = x
            best_z = z


    return best_article














