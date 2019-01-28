# Evaluating a Classification Model 

## Agenda
- What is the purpose of **model evaluation**, and what are some common evaluation procedures?
- What is the usage of **classification accuracy**, and what are its limitations?
- How does a **confusion matrix** describe the performance of a classifier?
- What **metrics** can be computed from a confusion matrix?
- How can you adjust classifier performance by **changing the classification threshold**?
- What is the purpose of an **ROC curve**?
- How does **Area Under the Curve (AUC)** differ from classification accuracy?

## Review of Model Evaluation
- Need a way of choosing between models: different model types, tuning parameters, and features
- Use a **model evaluation procedure** to estimate how well a model will generalize to out-of-sample data
- Requiares a **model evaluation metric** to quantify the model performance


### Model evaluation procedures

1. **Training and testing on the same data**
    - Rewards overly complex models that "overfit" the training data and won't necessarily generalize
2. **Train/test split**
    - Split the dataset into two pieces, so that the model can be trained and tested on different data
    - Better estimate of out-of-sample performance, but still a "high variance" estimate
    - Useful due to its speed, simplicity, and flexibility
3. **K-fold cross-validation**
    - Systematically create "K" train/test splits and average the results together
    - Even better estimate of out-of-sample performance
    - Runs "K" times slower than train/test split


### Model evaluation metrics
- **Regression problems:** Mean Absolute Error, Mean Squared Error, Root Mean Squared Error
- **Classification problems:** Classification accuracy

### Null Accuracy:
Anytime you are using **classification accuracy** as your evaluation metric, you should compare it against **null accuracy**. **Null Accuracy**: the accuracy that could be achieved by always predicing the most frequent class in the testing set. Null accuracy answers the question, "If my model was to predict the predominant class 100% of the time, how often would it be correct?"

Reason for the **Null Accuracy comparison**: 

Say you are training a model to predict the diabetes status of a patient given their health measurements. You use train/test split to split the data, use a logistic model to be trained on the training dataset and then make predictions on the testing dataset. Now calculate your classification accuracy using `metrics.accuracy_score(y_test, y_pred)` where `y_test` is the true response and `y_pred` is the prediction the trained model is estimating. You get 69% accuracy. That seems good, right? 

Now, use your **null model** as your baseline; *null will predict the most frequent class 100% of the time*. In your testing dataset, the most frequent class is 0 (patients without diabetes), which is 130 of the patients out of 192 patients. That means the **null accuracy** will predict the patient does not have diabetes ((130/192)x100%) = 68% of the time. Thus, a model that only predicts 1% better than the null is a VERY POOR MODEL!!! This demonstrats one weakness of classification accuracy as a model evaluation metric in that classifaction accuracy doesn't tell us anything about the underlying distribution of the testing set. 

#### The **null accuracy** for classification.
When there are only two classes, we use 0's and 1's to represent the classes. The following code works to calculate the **null accuracy** in these cases:
```python
# calculate null accuracy (for binary classification problems coded as 0/1)
max(y_test.mean(), 1 - y_test.mean())
```

Where there are more than 2 classes, thus we have 0's, 1's, 2's, 3's, etc... in the response vector, then the following code can be used to calculate the **null accuracy**:
```python
# calculate null accuracy (for multi-class classification problems)
y_test.value_counts().head(1) / len(y_test)
```

## Notes on Classification Metric
- It is the easity metric to understand 
- But, it does not tell you about the underlying distribution of the respone values, which we examine by calculating the **null accuracy**. 
- Also, it does not tell you the **"types" of errors** your model is making, which is often useful to know in real world situations. 


## Confusin Matrix:
*__Important Note:__ all scikit-learn metrics assume the true values vector is passed first, so get used to doing this!*
```python
# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y_test, y_pred_class))
```
|     	   | Predicted 0  | Predicted 1 |
| ------------------------------------- |
| Actual 0 | 	118		  | 	12		| 
| Actual 1 | 	47		  | 	15		| 





