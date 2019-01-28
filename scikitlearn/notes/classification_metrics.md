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
| ---------|--------------|------------ |
| Actual 0 | 	118		  | 	12		| 
| Actual 1 | 	47		  | 	15		| 


0 - does not have diabetes

1 - has diabetes


- So, when we predict that patient 1 has diabetes and they actually do, this is called a **True Positive**. 
- When we predict that patient 1 has diabetes and they do NOT actually have it, this is called a **False Positive**. We Falsely predicted a positive result. A way to recall this easily; assign 1 to a positive diagnoses and it's easier to remember as a positive result since 1 is a positive value. 
- When we predict patient 1 does NOT have diabetes and they do NOT have it in reality, this is called a **True Negative**. We truly (accurately) predicted the result was negative for diabetes. 
- When we predict patient 1 does NOT have diabetes and they actually have it, this is called a **False Negative**. We falsely predicted that they were negative for diabetes.

The *first term (True or False)* represents the accuracy of the user's guess to the true response. The *second term (Positive or Negative)* represents the guess made. 

In brief,
**Basic terminology**
- **True Positives (TP):** we *correctly* predicted that they *do* have diabetes
- **True Negatives (TN):** we *correctly* predicted that they *don't* have diabetes
- **False Positives (FP):** we *incorrectly* predicted that they *do* have diabetes (a "Type I error")
- **False Negatives (FN):** we *incorrectly* predicted that they *don't* have diabetes (a "Type II error")


Thus, the confusion matrix can be shown to represent:

|     	   | Predicted 0  | Predicted 1 |
| ---------|--------------|------------ |
| Actual 0 | # of TN 	  | # of FP 	| 
| Actual 1 | # of FN	  | # of TP		| 


### Metrics computed from a confusion matrix
The confusion matrix helps you to understand the performance of your classifier. But how can it help you to choose between models? It's not a model evaluation metric, so you can't simply tell scikit-learn to choose the model with the best confusion matrix. However, there are many metrics that can be calculated from a confusion matrix and those can be directly used to choose between models. 

First, we break up the confusion matrix into its constituents:
```python
confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1, 1]  # 15
TN = confusion[0, 0]  # 118
FP = confusion[0, 1]  # 12
FN = confusion[1, 0]  # 47
```

**Before answering any of the following questions, read the question, read over the Basic terminologies (definitions of TP, TN, FP, FN), then answer.**


#### 1. Classification Accuracy: How often is the classifier correct?
This would include all those in the confusion matrix starting with *True*; truthfully (accuratley) estimated. 
```python
print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, y_pred_class))
```


#### 2. Classification Error: How often is the classifier incorrect?
This would include all those in the confusion matrix starting with *False*; falsely estiamted. This is also referred to as the **misclassification rate**; the rate at which we misclassify the true response. 
```python
print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, y_pred_class))
```


#### 3. Sensitivity: When the actual value is positive, how often is the prediction correct?
How "sensitive" is the classifier to detecting positive instances? Sensitivity can also be referred to as the **true positive rate** or **recall**, depending on the area of study.

**Sensitivity is something you want to maximize**. 
Since 1 means a positive detection of diabetes, **sensitivity** will be high when your classifier accurately estimates 1 for a large number of patients who actually have diabetes. 
```python
print(TP / float(TP + FN))
print(metrics.recall_score(y_test, y_pred_class))
```
FN means we falsely (incorrectly) guessed (F) that they were negative (N) for diabetes, when they are actually positive for diabetes. This is why FN is a positive diabetes result. 

#### 4, Specificity: When the actual value is negative, how often is the prediction correct?
How "specific" (or "selective") is the classifier in predicting positive instances? 
**Specificity is also something you want to maximize, in addition to sensitivity**. 
```python
print(TN / float(TN + FP))
```
FP means we falsely (incorrectly) guessed (F) that they were positive (P) for diabetes, when they were actually negative for diabetes. This is why FP is a negative diabetes result. 


#### 5. False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
```python
print(FP / float(TN + FP))
``` 

#### 6. Precision: When a positive value is predicted, how often is the prediction correct?
How "precise" is the classifier when predicting positive instances?
```python
print(TP / float(TP + FP))
print(metrics.precision_score(y_test, y_pred_class))
```


### Conclusion:

- Confusion matrix gives you a **more complete picture** of how your classifier is performing
- Also allows you to compute various **classification metrics**, and these metrics can guide your model selection

**Which metrics should you focus on?**

- Choice of metric depends on your **business objective**
- **Spam filter** (positive class is "spam"): Optimize for **precision or specificity** because false negatives (spam goes to the inbox) are more acceptable than false positives (non-spam is caught by the spam filter)
- **Fraudulent transaction detector** (positive class is "fraud"): Optimize for **sensitivity** because false positives (normal transactions that are flagged as possible fraud) are more acceptable than false negatives (fraudulent transactions that are not detected)


