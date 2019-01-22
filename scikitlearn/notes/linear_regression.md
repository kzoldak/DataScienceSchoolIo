# Least Squares (LS)
Least squares is a standard approch in regression analysis for approximating the solution of overdetermined systems, i.e., sets of equations in which there are more equations than unknowns. The term "least squares" means that the overall solution minimizes the sum of squares of the residuals made in the results of every equation.

## Regression
Least squares' most frequently used application is in fitting models to data. When fitting data with a model, the best-fit model (and its parameters) occurs when the sum of the squared residuals is minimized. 

### Residuals
Geometrically, residuals are the vertical distances between the dependent variable's observations (the y variable) and its estimates as predicted by the model; or the difference between the data point's y-axis value and the y-model value at the same value of x. 

Residuals are also referred to as the errors in error analysis. The estimated model value for y (y-model) is always subtracted from the observed y value (y-data) so that positive residuals occur when the data point is located above the model and negative residuals occur when the data point is located below it. 

#### Residual Sum of Squares (RSS)
Total residuals for a given model is the "residual sum of squares" (RSS). 
Least squares line minimizes RSS. A lot of people simply refer to this as the "sum of squares" (SOS). sum( (y - ymodel)^2 ). 

### Cons for LS in Regression
When the independent variable (the x variable) has large uncertainties. 

## Two LS Categories
Least squares falls into two main categories; linear least squares (or ordinary least squares) and nonlinear least squares. When fitting a linear model to the data, if there is a pattern to the residuals, this is a sign that you should not be using a linear model to fit the data. 




# Ordinary Least Squares (OLS), AKA Linear Least Squares
Ordinary least squares (OLS) is a type of least squares method for estimating model parameters when a linear regression model is used. The mathematical format of the linear model would be: 
y = \beta_0 + (\beta_1 * x_1) + (\beta_2 * x_2) + ... + (\beta_n * x_n)
where 
- y is the response
- \beta_0 is the intercept
- \beta_1 is the coefficient for x_1 (the first feature)
- \beta_n is the coefficient for x_n (the nth feature)

If there is only one feature, then this equation reduces to the familar format of y = \beta_0 + (\beta_1 * x_1) or y = b + mx. 


## Simple Linear Regression
Simple linear regression is an approach for predicting a **quantitative response** using a **single feature** (or "predictor" or "input variable"). It takes the following form:

y = \beta_0 + (\beta_1 * x)

What does each term represent?
- y is the response
- x is the feature
- \beta_0 is the intercept (y-intercept)
- \beta_1 is the coefficient for x (slope)

Together, \beta_0 and \beta_1 are called the **model coefficients**. To create your model, you must "learn" the values of these coefficients. And once we've learned these coefficients, we can use the model to predict future observations. 


## Useful Terms
homoscedastic: *homo* (same) *skedasis* (dispersion) 
- in statistics, a sequence or a vector of random variables is homoscedastic if all radom variables in the sequence or vector have the same finite variance. AKA "homogeneity of variance". Another spelling for it is homoskedasticity.
heteroscedastic: *hetero* (different) *skedasis* (dispersion) 
- in statistics, a sequence or a vector of random variables is heteroscedastic if there are sub-populations within the dataset that have different variabilities. Thus, the variance in the data differs across x-y space. 


