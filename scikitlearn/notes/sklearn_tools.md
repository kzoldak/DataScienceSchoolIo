As I learn about scikit-learn, I record a *short* background on each of the algorithms I learn and give an example of a dataset where I think the algorithm would be useful. These examples are guesses and based on my scientific background. When I'm done learning the basics of these tools, this is a good place to start when performing short examples of each. 


# Classification Methods. Categorical.
---

## NearestNeighbors
`from sklearn.neighbors import NearestNeighbors`
* unsupervised learning with nearest neightbors. 
* simple task of finding the nearest neighbors between two sets of data.
https://scikit-learn.org/stable/modules/neighbors.html


## K Nearest Neighbors (KNeighborsClassifier)
`from sklearn.neighbors import KNeighborsClassifier`
* supervised learning with nearest neighbors classification. 
* finds nearest neighbors to an unknown observation and classifies that unknown observation based on the classification labels of those neighbors. 
* Uses training data and a respose.
* dataset: predicting GRBs that would have been detected by the LAT had they not been outside its field of view. Classes: GBM burst, GBM+LAT burst. 
* dataset: predicting AGN tye?  Star Cluster type?
https://scikit-learn.org/stable/modules/neighbors.html#classification

## Logistic Regression
`from sklearn.linear_model import LogisticRegression`
* is a linear model for classification rather than regression (despite its name). AKA logit regression in literature. 

In agriculture: modeling crop response
* The logistic S-curve can be used for modeling the crop response to changes in growth factors. There are two types of response functions: positive and negative growth curves. For example, the crop yield may increase with increasing value of the growth factor up to a certain level (positive function), or it may decrease with increasing growth factor values (negative function owing to a negative growth factor), which situation requires an inverted S-curve.



