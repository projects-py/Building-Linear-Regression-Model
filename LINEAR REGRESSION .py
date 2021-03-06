#importing libraries that we need
import pandas as pd
import math
#importing library and later on we will import dataset from sklearn.datasets
from sklearn import datasets




#dataset
dataset = datasets.load_diabetes()



#printing description of dataset
print(dataset.DESCR)


# # Feature Names

print(dataset.feature_names)


# # Creating X and Y data matrices


X = dataset.data
Y = dataset.target




print(X.shape)
print(Y.shape)


# # Data Split

from sklearn.model_selection import train_test_split


# # Dividing dataset into 80/20 for training and testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# # Dimension for Training and Testing data

X_train.shape, Y_train.shape

X_test.shape, Y_test.shape


# ## Building a Linear Regression Model

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


model = linear_model.LinearRegression()


# # Training model

model.fit(X_train, Y_train)


# # Predicting

Y_pred = model.predict(X_test)


# # Prediction Results

# Print model performance


print('Coefficients: ', model.coef_)
print("Intercept: ", model.intercept_)
print("MSE: %.2f" %  mean_squared_error(Y_test, Y_pred))
print("Root Mean Squared Error: ", math.sqrt(mean_squared_error(Y_test, Y_pred)))
print("Coeff of determination: %.2f" % r2_score(Y_test, Y_pred))



print(Y_test)

print(Y_pred)


# # Scatter Plots

import seaborn as sns
sns.scatterplot(Y_test,Y_pred, marker = '+')


# # Inference

#As we can see that this model has very low accuracy as the training data is not big enough
