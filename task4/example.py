import numpy as np
from time import sleep

X = {}
B = {}
b = {}
A = {}

d=36
k=6
alpha=1

best_article = -1
best_x_t_a = -1
best_z_t_a = -1

def set_articles(articles):
    '''
    Save the articles globally as a hashmap and initialize for hybrid LinUCB

    Arguments:
    articles - a hashmap of all articles
    '''
    print ""
    print "set articles"
    print "============"

    global X
    X = articles

    # Initialize A_0 and b_0
    global A
    global b
    A[0] = np.identity(k)
    b[0] = np.zeros([k,1])

def update(reward):
    '''
    Update the weights
    '''
    print ""
    print "update"
    print "======"

    # Check if the reward is positive
    if reward == -1:
        return

    # Set all the variables to global
    global A
    global B
    global b
    global X
    global best_article
    global best_score
    global best_x_t_a
    global best_z_t_a

    # Update the weights
    B_a = B[best_article]
    B_a_T = np.transpose(B_a)
    A_a = A[best_article]
    A_a_inv = np.linalg.inv(A_a)
    x_t_a = best_x_t_a
    z_t_a = best_z_t_a
    
    A[0] += B_a_T * A_a_inv * B_a
    b[0] += B_a_T * A_a_inv * b[best_article]
    A[best_article] += x_t_a * np.transpose(x_t_a)
    B[best_article] += x_t_a * np.transpose(z_t_a)
    b[best_article] += reward * x_t_a
    A[0] += z_t_a * np.transpose(z_t_a) - np.inverse(B[best_article]) * np.inverse(A[best_article]) * B[best_article]
    b[0] += reward * z_t_a - np.inverse(B[best_article]) * np.inverse(A[best_article]) * b[best_article]

def recommend(time, user_features, choices):
    print ""
    print "recommend"
    print "========="

    # Set all the variables to global
    global A
    global B
    global b
    global X
    global best_article
    global best_score
    global best_x_t_a
    global best_z_t_a

    z = user_features.reshape([k,1])
    z_T = np.transpose(user_features)

    # Calculate beta_hat
    A_0_inv = np.linalg.inv(A[0])
    beta_hat = A_0_inv*b[0]
    
    # Estimate the score per article
    best_article = -1
    best_score = -1
    for article in choices:
        if not article in A:
            A[article] = np.identity(d)
            B[article] = np.zeros([d,k])
            b[article] = np.zeros([d,1])
            
        # Calculate x_t_a
        x = X[article].reshape((k,1))
        x_t_a = (x * np.transpose(z)).ravel()

B_a = B[article]
B_a_T = np.transpose(B[article])

        # Estimate our phi_a_hat
        A_a_inv = np.linalg.inv(A[article])
        phi_a_hat = A_a_inv * (b[article] - B[article] * beta_hat)
        
        # Calculate the variance s_t_a
        first_term = z_T * A_0_inv * z
        second_term = 2 * z_T * A_0_inv * B_a_T* A_a_inv * x_t_a
        third_term = np.transpose(x_t_a) * A_a_inv * x_t_a
        forth_term = np.transpose(x_t_a) * A_a_inv * B_a * A_0_inv * B_a_T * A_a_inv * x_t_a

        s_t_a = first_term - second_term + forth_term + third_term
        
        p_t_a = z_T * beta_hat + np.transpose(x_t_a) * phi_a_hat + alpha * np.sqrt(s_t_a * 1.0)

        if p_t_a > best_score:
            best_article = article
            best_score = p_t_a
            best_x_t_a = x_t_a
            best_z_t_a = z

    return best_article














