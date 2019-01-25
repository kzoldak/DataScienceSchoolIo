# Algorithm Types
1. Classification
2. Regression
3. Clustering
4. Dimensionality Reduction

# 1. Classification
- Predicting a categorical response 

## Model Evaluation Metrics
- **accuracy**


# 2. Regression
- Predicting a continuous response

## Model Evaluation Metrics
- **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
	- `metrics.mean_absolute_error(true, pred)`
$$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$

- **Mean Squared Error** (MSE) is the mean of the squared errors:
	- `metrics.mean_squared_error(true, pred)`
$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$

- **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
	- `np.sqrt(metrics.mean_squared_error(true, pred))`
$$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
