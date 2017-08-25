# Task 4: Exploring bandits using LinUCB

For this task we have implemented the LinUCB algorithm as introduced in the lecture.
The recommend function loops through all the choices and adds articles not yet seen to our hashmaps M, b and M_inv.

For each article the UCB and subsequently the expected payoff is calculated. The article with the highest expected payoff is chosen.

The update()-method simply updates our matrix M and vector b. As the matrix M is only mutated in the update()-method, we decided to move the calculation of the inverse matrix M^-1 there as well. (But as far as we know, LinUCB was far from reaching the runtime limit of 30 minutes, so this improvement was not that important.)

We also scaled the reward factors, so instead of using the passed 0 and 1 rewards, we tried to choose different values. For searching for the optimal alpha, positive and negative reward, we conducted experiments on the Euler cluster.

PS: We also tried to implement Hybrid LinUCB, but got for the very same submission sometimes a timeout and sometimes not. It seems that the runtime limit does not depend solely on the runtime of our submission, but also on the load of the server. This makes the whole submission highly unpredictable. Thus we have decided to stick with the simpler LinUCB model.