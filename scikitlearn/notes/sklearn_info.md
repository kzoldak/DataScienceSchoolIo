Some important information for using scikit-learn.



Requirements for working with data in scikit-learn:

 
1. Features and response are separate objects
	- The features matrix (or 2D array) will be the training data. This matrix will have dimensions of nxp, where n is the number of observations and p is the number of features. p can be changed by selecting only certain columns of a dataframe. Essentially, n is the number of rows and p is the number of columns in the data frame that is used to represent the training data. 
	- Features is typically represented by a capital X. 
	- Response is a 1D array holding numeric values representing the response. It should be n rows long and 1 column wide. 
	- all values in the features matrix and the response array MUST be numeric values, not strings. 

2. Features and response should be numeric
	- no strings anywhere in either.
3. Features and response should be NumPy arrays
	- np.ndarray or np.array or np.asarray
4. Features and response should have specific shapes
	- see # 1.
5.  Both the feature data (training data) and the response data (what you are trying to predict) must be passed to the scikit-learn model.


"Instantiate" the "estimator"
- "Estimator" is scikit-learn's term for model
- "Instantiate" means "make an instance of"





How do we 
1. choose which model to use for my supervised learning task.
2. choose best tuning parameters for that model.
3. estimate the likely performance of my model on out of sample data.

The goal of supervised learning is always to build a model that generalizes to out of sample data. 
We need a procedure that allows us to estimates how well a given model is likely to perform on out of sample data. This is known as a Model evalutioan procedure. Likely performance of all 3 models. Use that performance estimate to choose between the models. 


# Determining accuracy of the models
## Train and Test (same dataset)
NOT A GOOD OPTION FOR DETERMINING BEST MODEL, BUT NEEDS TO BE DISCUSSED. 
Train and test the same dataset. 
IDEA: Train our model on the entire dataset and then test our model by checking how well it performs on that same dataset. If we use a dataset that we know the response for, we can check how well our model is doing by comparing the predicted response values with the true response values. 
Why is this a bad idea? 
This is considered maximizing the training accuracy, which is essentailly overfitting the data. Overfitting the data leads to our model learning the NOISE in our data instead of the signal. Thus, the future predictions using this model will not be good at predicting NEW observations, only the old ones with its noise. We have created an overly complex model for the training data and not a generalized model for predicting future observations (out of sample observations).
In k nearest neighbors, k=1 is overfitting. By definition of the algorithm, the knn algorithm using k=1 (1 neighbor) will always return a 100% accuracy when making predictions with a model that was trained on the exact same dataset. 

### command for finding the percentage of predictions that match the original response data.
metrics.accuracy_score(y, y_pred)


Training and testing your models on the SAME dataset is not useful for deciding which model to choose. What else can we use?
## Train/Test Split
#### AKA Test Set Approach, Validation Approach. May be slightly different from his example. 
STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split




