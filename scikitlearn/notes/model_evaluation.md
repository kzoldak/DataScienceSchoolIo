# Model Evaluation (or Cross Validation)
__Goal:__ 
- How do I choose which model to use for my supervised learning task?
- How do I choose the best tuning parameters for that model?
- How do I estimate the likely performance of my model on out-of-sample data?


# What is Model Evaluation, or Cross Validation?
- Sometimes called rotation estimation. 
- Cross validation is a method for model validation; assessing how the results of a statistical analysis will generalize to out-of-sample data. Thus, it is a method for out-of-sample testing. 
- It is used to determine how well a predictive model will perform in practice. 
- The goal of cross-validation is to test the model’s ability to predict new data that was not used in estimating it, in order to flag problems like overfitting or selection bias. [Cawley, Gavin C.; Talbot, Nicola L. C. (2010). "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation" (PDF). 11. Journal of Machine Learning Research: 2079–2107.](http://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)
- k-fold cross-validation




## Evaluation Procedures (Cross Validation Procedures):
1. Train/Test Split
2. Cross Validation 


## Notes:
From [scikit-learn documentation](https://scikit-learn.org/stable/modules/cross_validation.html). Learning the parameters of a model by testing it on the same exact data set that was used to train the model is poor practice. The model essentially learns the random fluctuations in the data, and tries to fit that instead of being able to generalize to new data observations. It would be capable of predicting its own data with very high accuracy, but it wouldn't help with any new observations that are "out-of-sample". This is called **overfitting** the training data. Overfitting creates unnecessarily complex models, which is rewarded by this maximization of the **training accuracy**, and not the testing accuracy. Such a model is not capable of generalizing to future observations, and thus is useless to us. 

One way we can avoid overfitting the training data is to split up the data set into subsets of training and testing data. Therefore the model is trained on only the training data and tested on only the testing data. The testing data can be though of as new observations, or out-of-sample observations. The **train/test split** procedure performs just this. 





# Cross Validation Procedures

## Train/Test Split
- Data is split into 2 parts; one part is the **training set** and the other is the **testing set**. 
- You traing the model on the training set and test the model on the testing set. 
- There is no general rule for the proportionality of this split, but people tend to use anywhere from 20-40% of their dataset's observations for testing the model. This means the remainder of the dataset is used for training the model. 
- `scikit-learn`'s `train_test_split` function has an optional `test_size` parameter that tells the function what percentage of data to use for the testing dataset. 
- Since model was technically trained and tested on different data, and we know the response values for both the training and the testing data, we can test our model's accuracy for predicting out-of-sample responses. This is done by using `scikit-learn`'s `metrics.accuracy_score(y_test, y_pred)` function, where y_test are the true responses (of the testing dataset) and y_pred is the model's prediction of those responses, based on the training dataset's features matrix and response vector. 
- This procedure works because as far as our model knows, the testing dataset is out-of-sample data. At the time the model was trained, it had no knowledge of the testing dataset's existence. 
- **Testing accuracy** is a better estimate of out-of-sample data than **training accuracy**. 

## Advantages to Train/Test Split
- It produces **testing accuracy** and not **training accuracy**, so it penalizes models that are too complex and overfitting the data. Testing accuracy also penalizes on models that are not complex enough. 
- Flexible and fast.


## Disadvantages of Train/Test Split
- Provides a **high-variance estimate** of out-of-sample data. **K-fold cross-validation** overcomes this limitation. 


### Train/Test Split's Advantages over K-fold Cross Validation
- runs K times faster than k-fold cross-validation. 
- simpler to examine the detailed results of the testing procedure. 





## K-fold Cross Validation
- Splits the dataset into K equal partitions (aka "folds"). One partition's data is used to test the model while all the rest of the partition's data are combined and used to train the model. If there are 5 partitions, each time this is done, one of the 5 partitions is left out of the training data and used to test the model's accuracy. Thus, there are K iterations of this process. After each of the K iterations, a testing accuracy is calculated, resulting in K total testing accuracies. The average of those K testing accuracies is used to estimate the out-of-sample accuracy of the model. 
- K-fold Cross Validation is a *training accuracy* method. These types of methods avoid overfitting, however, it is a high variance estimate of out of sample accuracy. The testing accuracy can change a lot depending on which observations happen to be in the testing set. 

### Recommendations for its use
- K=10 is recommended; typically gives best out of sample accuracy. This has been tested and published. 
- For *classification* problems, **stratified sampling** is recommended for creating the folds. `Scikit-learn`'s `cross_val_score` function automatically uses **stratified sampling**. **Stratified sampling** is when each response class is represented with equal proportions in each of the k-folds. For example, if you have a dataset where 20% of the data is classified as `BLUE`, then each of the k-folds should also contain data where 20% of the data are classified as `BLUE`. Equal ratio proportionalities between folds. 

### Parameter Tuning
- selecting the best tuning parameters (aka hyper parameters) for the knn classification model. We want to choose the best tuning parameters that will produce a model that best generalizes to out of sample data. 

### K-fold Cross Validation's Advantages over Train/Test Split
- K-fold cv produces a more accurate estimate for out of sample data.
- K-fold is a more "efficient" use of data. Every observation is used for both training and testing the model.  



