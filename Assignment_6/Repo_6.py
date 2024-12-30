#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target


# In[6]:


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# In[8]:


# Predict
y_pred = model.predict(X_test)


# In[9]:


# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)


# In[10]:


# Feature Importance
coefficients = pd.Series(model.coef_, index=X.columns)
coefficients.sort_values(ascending=False).plot(kind='bar', title='Feature Importance', figsize=(10, 6))
plt.ylabel('Coefficient Value')
plt.show()


# In[ ]:




