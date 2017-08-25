# Task 3: Coreset-Sampling

Mapper: To generate the coresets we implement importance sampling on the mapper. The bicriterion points for this are sampled by the D^2 probabilities.
D^2: The first sample was chosen at random. All the subsequent bicriterion point were drawn from a weighted probability distribution corresponding to the squared distance of the points to the already chosen ones. As D^2 sampling is expensive we kept the sample size at 20.
Importance sampling: We loop twice over our data points. Once to attach each data point to its nearest bicriterionPoint and in the second pass to calculate sampling probabilities for each point. After that we draw 900 points from the calculated distribution. These points comprise the coreset.

In addition we not only pass the points that the corest comprises off to the reducer, but also the weights of those points, which is the inverse of the sampling distribution.

Reducer: Here we decided to merge the coresets without compressing them. After merging (, which is done by the reducer automatically), we run a weighted KMeans algorithm on the merged points. In our experiments and tests the simple Lloyds algorithm proved to give the best results for us.