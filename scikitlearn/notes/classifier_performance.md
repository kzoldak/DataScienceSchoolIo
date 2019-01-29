# Classifier Performance
**How to modify the performance of a classifier by adjusting the classification threshold. _Read classification_metrics.md before this._**

```python
# print the first 10 predicted responses
logreg.predict(X_test)[0:10]
# out:  array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1])

# print the first 10 predicted probabilities of class membership
logreg.predict_proba(X_test)[0:10, :]
# out: array([[0.63247571, 0.36752429],
#			  [0.71643656, 0.28356344],
#			  [0.71104114, 0.28895886],
#			  [0.5858938 , 0.4141062 ],
#			  [0.84103973, 0.15896027],
#		      [0.82934844, 0.17065156],
#			  [0.50110974, 0.49889026],
#			  [0.48658459, 0.51341541],
#			  [0.72321388, 0.27678612],
#			  [0.32810562, 0.67189438]])
```
The 0th column would be the probabilities of each patient getting a 0 (negative for diabetes) and the second column would be the probabilities of the same patients getting a 1 (positive for diabetes). 

__A threshold of 0.5:__ Above 0.5 in the 0th column means negative for diabetes. Above 0.5 in the 1st column means positive for diabetes. Thus, we would get `[0, 0, 0, 0, 0, 0, 0, 1, 0, 1]` with this data, which is the result of `logreg.predict(X_test)[0:10]`. However, what if we wanted to change this threshold from 0.5 to something else?


We want to **decrease the threshold** for predicting diabetes in order to **increase the sensitivity** of the classifier.
This can be done with `binarize`, but first we need to store all the probabilities for the 1's (diabetes positive) into an array called `y_pred_prob`:

```python
# store the predicted probabilities for class 1
y_pred_prob = logreg.predict_proba(X_test)[:, 1] # 
```
Now we will change the threshold to 0.3:
```python
# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize
y_pred_class = binarize([y_pred_prob], 0.3)[0]

# print again the first 10 predicted probabilities 
y_pred_prob[0:10]
# out: array([0.36752429, 0.28356344, 0.28895886, 0.4141062 , 0.15896027,
#             0.17065156, 0.49889026, 0.51341541, 0.27678612, 0.67189438])


# print the first 10 predicted classes with the lower threshold
y_pred_class[0:10]
# out: array([1., 0., 0., 1., 0., 0., 1., 1., 0., 1.])
```
Notice how the output of `y_pred_class` now reflects the probablities in `y_pred_prob` with the new threshold of 0.3 in mind. Anything with a probability of 0.3 or greater is now considered a positive diabetes result. 
We have **increased the sensitivity of our classifier** by decreasing the threshold to 0.3 for diabetes detections. Our model is now more sensitive to positive instances. All observations with predicted probabilities > 0.3 will be a potitive instance (positive for diabetes). Recall we are trying to predict the change that someone will end up with diabetes, based on certain features of their lifestyle. This does not mean that someone will absolutely end up with the disease just becuase they get a 1. If they get a value of 1 and their probability is fairly high, we'd want to contact those patients and warn them that their probability of getting diabetes is quite high. 

#### Compare Confusion Matrices, using threshold of 0.5 vs 0.3
```python
# previous confusion matrix (default threshold of 0.5)
print(confusion)
# out: [[118  12]
#       [ 47  15]]

# new confusion matrix (threshold of 0.3)
print(metrics.confusion_matrix(y_test, y_pred_class))
# out: [[80 50]
#       [16 46]]
```
Our sensitivity has increased from 0.24 to 0.74 and our specificity has decreased from 0.91 to 0.62. 


__Sensitivity:__ When the actual value is positive, how often is the prediction correct?

	TP / float(TP + FN)
	metrics.recall_score(y_test, y_pred_class)

__Specificity:__ When the actual value is negative, how often is the prediction correct?

	TN / float(TN + FP)


```python
# Sensitivity Calculation: 
# print(TP / float(TP + FN))
print(46 / float(46 + 16))
print(metrics.recall_score(y_test, y_pred_class))
# out: 0.7419354838709677

# Specificity Calculation:
# TN / float(TN + FP)
print(80 / float(80 + 50))
# out: 0.6153846153846154
```

**Conclusion:**

- **Threshold of 0.5** is used by default (for binary problems) to convert predicted probabilities into class predictions
- Threshold can be **adjusted** to increase sensitivity or specificity
- Sensitivity and specificity have an **inverse relationship**



