#!/usr/bin/env python
# coding: utf-8

# # Building linear regression model 

# In[82]:


#importing libraries that we need
import pandas as pd
import math
#importing library and later on we will import dataset from sklearn.datasets
from sklearn import datasets


# In[83]:


#dataset
dataset = datasets.load_diabetes()


# In[84]:


#printing description of dataset
print(dataset.DESCR)


# # Feature Names

# In[85]:


print(dataset.feature_names)


# # Creating X and Y data matrices

# In[86]:


X = dataset.data
Y = dataset.target


# In[87]:


print(X.shape)
print(Y.shape)


# # Data Split

# In[88]:


from sklearn.model_selection import train_test_split


# # Dividing dataset into 80/20 for training and testing

# In[89]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# # Dimension for Training and Testing data

# In[90]:


X_train.shape, Y_train.shape


# In[91]:


X_test.shape, Y_test.shape


# ## Building a Linear Regression Model

# In[92]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[93]:


model = linear_model.LinearRegression()


# # Training model

# In[94]:


model.fit(X_train, Y_train)


# # Predicting

# In[95]:


Y_pred = model.predict(X_test)


# # Prediction Results

# Print model performance

# In[96]:


print('Coefficients: ', model.coef_)
print("Intercept: ", model.intercept_)
print("MSE: %.2f" %  mean_squared_error(Y_test, Y_pred))
print("Root Mean Squared Error: ", math.sqrt(mean_squared_error(Y_test, Y_pred)))
print("Coeff of determination: %.2f" % r2_score(Y_test, Y_pred))


# In[97]:


print(Y_test)


# In[98]:


print(Y_pred)


# # Scatter Plots

# In[99]:


import seaborn as sns
sns.scatterplot(Y_test,Y_pred, marker = '+')


# # Inference

# In[103]:


#As we can see that this model has very low accuracy as the training data is not big enough


# In[ ]:




