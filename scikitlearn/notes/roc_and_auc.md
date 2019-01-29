__*Read both classification_metrics.md and classifier_performance.md before this one, in that order.*__

# ROC Curves and Area Under the Curve (AUC)

## ROC Curves
It is incredibly inefficient to search for an optimal threshold value (such as 0.5 of 0.3 from the last file) by trying different threshold values one at a time. What if we could see how **sensitivity** and **specificity** are affected by various thresholds, without actually having to try each threshold?
**PLOT THE ROC CURVE!**
```python
# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
```
You want to balance **sensitivity** and **specificity** when choosing a threshold. This plot allows you to visualize that balance and choose the best threshold for your problem. The upper-left corner is the sweet spot becuase you want a *high sensitivity* and *low specificity*. Unfortunately, you can not see the actual thresholds used to generate the ROC curve on the curve itself. However, you can write a function that allows you to pass it a threshold value and see the resulting sensitivity and specificity. 
```python
# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])

evaluate_threshold(0.5)
# out:
# Sensitivity: 0.24193548387096775
# Specificity: 0.9076923076923077

evaluate_threshold(0.3)
# out:
# Sensitivity: 0.7258064516129032
# Specificity: 0.6153846153846154
```

## Area Under the Curve (AUC)
AUC is the **percentage** of the ROC plot that is **underneath the ROC curve**. An ideal classifier will hug the upper-left hand corner of an ROC plot and as a result, maximizes the area located under it on the plot. Thus, an ideal classifier will have a large AUC value. A higher AUC value is indicitive of a better overall classifier. AUC is often used as a single number summary of the performance of a classifier as an alternative to classificaiton accuracy. 

### AUC Calculation
```python
# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, y_pred_prob))
# out: 0.7245657568238213

```
- AUC is useful as a **single number summary** of classifier performance.
- If you randomly chose one positive and one negative observation, AUC represents the likelihood that your classifier will assign a **higher predicted probability** to the positive observation.
- AUC is useful even when there is **high class imbalance** (meaning on of the classes dominates), unlike classification accuracy. An example of a high class imbalance would be fraudulent bank transactions verses valid transactions. We expect very few fraud transactions. 
- Because AUC is useful as a metric for choosing between models, it is available as a scoring function for `cross_val_score`:
```python
# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
# out: 0.7378233618233618
```


# Video Conclusions:
- discussed many ways to evaluate a classifier. 
- Confusion Matrix and ROC/AUC are useful tools that describe how your classifier is performing. Suggest to use both of them whenever possible. 

## Confusion Matrix Advantages:
- Allows you to calculate a **variety of metrics**. You can focus on the metrics that match your problem. 
- Useful for **multi-class problems** (more than 2 response classes).

## ROC Curves and AUC Advantages:
- Does not require you to set a classification threshold, unlike the confusion matrix. 
- Still useful when there is high class imbalance. However, they are less interpretable than the confusion matrix for multi-class problems. 






