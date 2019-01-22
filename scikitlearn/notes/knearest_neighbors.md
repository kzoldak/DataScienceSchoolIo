# K Nearest Neighbors (KNN)

-----------

__!!! WARNING !!!:__ 

A very low value of k (number of neighbors) overfits the data. It follows the noise in the data rater than the signal. Using a low k creates an overly complex model that overfits the data. 


__[HERE](http://scott.fortmann-roe.com/docs/BiasVariance.html) is a great discussion on the bias-variance tradeoff when using the k-nearest neighbors algorithm.__ 



## Main idea of K Nearest Neighbors:
An unknown observations is classified based on the k observations in the training set that are nearest to it. For example, if k=5 is chosen, then the model searches for the 5 observations in the training data set that are nearest to the unknown observation. The model calculates the numerical distance between the unknown observation and each individual observation of all n observations in the entire training data set. The distance metric is often `euclidean`, but this can be changed. 
The responses of the five nearest points (when k=n) to the unknown observation will be tallied and the most common value is used as the response for the unknown observation. For the Iris data set, the training data is a matrix with a shape (n x p) where n = 150 (observations) and p = 4 (features; 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'). These are the 4 columns (aka parameters aka features) of the training data set. The response is an array of shape (n,) where n = 150 observations. This 1D array holds the 0's , 1's and 2's, each representing an individual iris species: {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}. 

Imagine the 5 nearest neighbors to an unknown observation are: [0, 1, 1, 0, 1].  The response (species of the unknown iris observations) of the unknown observation will be asigned a value of 1 (or the Iris-versicolor species). The K-nearest neighbor classifier is a very simple method that works surprisingly well on many problems. 

KNN is a very simple machine learning model, but it can make highly accurate predictions if the different classes in the dataset have very dissimilar feature values. 

X - features (training) data, a 2D numpy array. 
y - response data, a 1D numpy array. 

Model learns the relationship between X and y, between the feature and response. 



## Notes:
K-nearest Neighbors classification is a supervised method because we have a set of known classifications and we would like to find the best classification to assign to our unknown observation. 

Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point. 



For the Penn State Statistics Summer School (2015), K-nearest Neighbors was covered on Wednesday and is stored in the stat_learning.pdf file. 





## In R program:
knn(training.set, test.set, training.set.labels, K) in class package. 


## In Python:
scikit-learn implements two different nearest neighbors classifiers: KNeighborsClassifier implements learning based on the k nearest neighbors of each query point, where k is an integer value specified by the user. RadiusNeighborsClassifier implements learning based on the number of neighbors within a fixed radius r of each training point, where r is a floating-point value specified by the user.
The k-neighbors classification in KNeighborsClassifier is the most commonly used technique. The optimal choice of the value k is highly data-dependent: in general a larger k suppresses the effects of noise, but makes the classification boundaries less distinct.
In cases where the data is not uniformly sampled, radius-based neighbors classification in RadiusNeighborsClassifier can be a better choice. The user specifies a fixed radius r, such that points in sparser neighborhoods use fewer nearest neighbors for the classification. For high-dimensional parameter spaces, this method becomes less effective due to the so-called “curse of dimensionality”.
The basic nearest neighbors classification uses uniform weights: that is, the value assigned to a query point is computed from a simple majority vote of the nearest neighbors. Under some circumstances, it is better to weight the neighbors such that nearer neighbors contribute more to the fit. This can be accomplished through the weights keyword. The default value, weights = 'uniform', assigns uniform weights to each neighbor. weights = 'distance' assigns weights proportional to the inverse of the distance from the query point. Alternatively, a user-defined function of the distance can be supplied to compute the weights.
