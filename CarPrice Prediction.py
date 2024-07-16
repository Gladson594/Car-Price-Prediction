#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[5]:


pf = pd.read_csv('CarPrice.csv')
pf.head()


# In[6]:


x = pf['horsepower']
y = pf['price']


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=30,test_size=0.3)


# In[8]:


model = LinearRegression()


# In[11]:


x_train = np.array(x_train).reshape(-1, 1)
model.fit(x_train,y_train)


# In[12]:


print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Training score:", model.score(x_train, y_train))


# In[14]:


x_test = np.array(x_test).reshape(-1,1)


# In[15]:


y_pred = model.predict(x_test)


# In[16]:


mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)


# In[17]:


print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2 score:",r2)


# In[18]:


plt.scatter(y_test, y_test ,color = 'blue',label = 'actual price')
plt.scatter(y_test, y_pred ,color = 'green',label = 'predicted price')
plt.xlabel("Actual price")
plt.ylabel("predicted price")
plt.title("Actual price vs predicted price")
plt.legend()
plt.show()


# In[ ]:




