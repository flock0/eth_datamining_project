# Task 2: Distributed Stochastic Gradient Descent using PEGASOS and MapReduce

Feature Transformation: Our transformation implements random fourier features (just as discussed in the lecture) with RBF kernel. Usually we would choose the number of drawn features to be smaller than the number of our original features, but we found that drawing more features than in the original dataset creates better results. In addition we have defined a gamma parameter that spreads out the sampled omegas.

Gradient Descent (Mapper): We use a slightly altered Mini-Batch PEGASOS-Algorithm for gradient descent. Our method preserves the momentum of the gradient in the previous step. The alpha-parameter is used to determine how much momentum is "carried-over" from the last step. For the gradient calculation we use standard Hinge-loss. The hyper parameters have been manually tuned.

Combination (Reducer): The mappers all solve a local stochastic gradient descent problem and output a weight vector. We have seen that simply averaging the results leads to reasonably well results. Thus the single reducer only calculates the average of the weight vector values he receives.