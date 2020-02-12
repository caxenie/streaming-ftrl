# Streaming FTRL

Streaming Follow-The-Regularized-Leader (FTRL) reference implementation for Generalized Linear Model (GLM).

Using the Online Gradient Descent (OGD) paradigm, Follow-the-Regularized-Leader (FTRL) is a natural modification of the basic Follow-the-Leader (FTL) algorithm in which we minimize the loss on all past rounds plus a regularization term. The goal of the regularization term is to stabilize the solution.

Implementation based on: H. B. McMahan, G. Holt, D. Sculley, et al. "Ad click prediction: a view from the trenches". In: The 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD 2013. 

### Instantiation:
This reference implementation is done for a Generalized Linear Model (GLM), basically a Logistic Regressor Classifier. For learning at massive scale, online algorithms for GLM (e.g., Logistic Regressor Classifier) have many advantages. Although the feature vector x might have billions of dimensions, typically each instance will have only hundreds of nonzero values. This enables efficient training on large data streams, since each training example only needs to be considered once. In many domains with high dimensional data, the vast majority of features are extremely rare. In fact, in some of the models, half the unique features occur only once in the entire training set of billions of examples. 
