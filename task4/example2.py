import numpy as np
from time import sleep

M = {}
M_inv = {}
b = {}
w = {}
X = {}

d = 6
alpha = 0.012
factor = 1
print "alpha:", alpha

last_article = -1
last_z = -1

def set_articles(articles):
    '''
    Save the articles globally as a hashmap and initialize for hybrid LinUCB

    Arguments:
    articles - a hashmap of all articles
    '''
    X = articles

def update(reward):
    '''
    Update the weights
    '''

    # Check if reward is positive
    if reward == -1:
        return

    # Set all variable to global
    global last_z
    global last_article
    global M
    global M_inv
    global w
    global b

    article = last_article
    z = last_z

    # Update the parameter
    M[article] += np.outer(z.ravel(),z.ravel())
    b[article] += factor * reward * z
    M_inv[article] = np.linalg.inv(M[article])
    w[article] = M_inv[article].dot(b[article])


def recommend(time, user_features, choices):
    z = np.array(user_features).reshape([d,1])
    z_T = np.transpose(z)

    # Set all variable to global
    global last_z
    global last_article
    global M
    global M_inv
    global w
    global b

    last_z = z

    best_score = -1
    best_article = -1
    
    for article in choices:
        # Check if the article is new
        if not article in M:
            # Initialize M,b,w,M_inv for our article
            M[article] = np.identity(d)
            b[article] = np.zeros([d,1])
            w[article] = np.zeros([d,1])
            M_inv[article] = np.identity(d)

            # Ensure that each article has been explored at least one time
            last_article = article
            break

        # Calculate the UCB
        variance = z_T.dot(M_inv[article]).dot(z)
        score = np.transpose(w[article]).dot(z) + alpha * np.sqrt(variance * 1.0)

        # Update the best article
        if score > best_score:
            best_score = score
            best_article = article

    last_article = best_article

    return best_article














