# Box-Jenkins
Time series algorithm that gives the optimal order of an ARIMA model.

Let's describe the object and methods in the class:

Object: tsa_econometrics: The user inserts the variable and the type of time serie (constant and trend: "ct", constant: "c" or no constant: "nc").

Method 1 (level_results): Shows the level results of the dickey fuller stationarity test and returns it with the converted variable (first differences, second differences). Additionally it will show the user the integration order of the variable.

Method 2 (plotting): This plots the Autocorrelation function and the Partial Autocorrelation function of the converted variable. The user needs to analyse this plots as well.

Method 3 (selection_criteria): The user inserts the level of significance and the results will be the optimal lags of the ARIMA model. After this the user could choose any of them.

Method 4 (resid_diag): The users writes the order of the ARIMA model: (ar,i,ma). This will show the user the simple plot of the residuals as well as the histogram (with the density function). Additionally it will print the maximum and the minimum values that cause the no - normality problem in the model.

Method 5 (fit_univariate): This will fit the model with the order determined in "method 4". The method also requires, whether or not, if a dummy variable dummy is going to be inserted and correct the no - normality problem, let's describe this options:

  1. insert_dummy: if True then it will require a maximum or minimum based dummy.
  2. maximum: if True, a dummy will be inserted based on the maximum residual that causes the no - normality problem.
  3. minimum: if True, a dummy will be inserted based on the minimum residual that causes the no - normality problem.
  4. Note: User needs to specify -> If maximum: True, then minimum: False and viceversa. If insert_dummy: False, then all the arguments are False

On the other hand, in this method, the user specifies the lbox test lags to analyse if the residuals are a random - walk type, with the null hypothesis: the residuals are distributed normally.

Method 6 (plot_predictions): This plots the predictions inside the sample.

Method 7 (future_predict): This plots the future predictions (out of sample). In order to do that, the user need to specify the future steps.

References:
N. Gujarati, D. and C. Porter, D. (2008). "Basic Econometrics" (pp. 777-782). McGraw-Hill.
