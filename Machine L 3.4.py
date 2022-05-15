#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn


# In[4]:


data=pd.read_csv('airbnb_listing_train.csv')
data.head()


# In[5]:


data.shape


# In[6]:


data.columns


# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


o = data.dtypes[data.dtypes == 'object'].index

data[o].describe()


# In[10]:


data.nunique()


# In[11]:


data.isnull().sum()


# # Treating the missing values
# 

# In[14]:


data.drop('neighbourhood_group', axis=1, inplace=True)


# In[15]:


data.head()


# In[16]:


data['name'].isnull().sum()


# In[17]:


n = data.loc[pd.isna(data['name']), :].index
data.loc[n]


# In[18]:


data['name'].value_counts()


# In[19]:


data['name'].mode()


# In[20]:


# impute with mode of name

data['name'] = data['name'].fillna(data['name'].mode()[0])


# In[21]:


data['last_review'].value_counts()


# In[22]:


data['last_review'].mode()


# In[23]:


# impute with mode of last_review

data['last_review'] = data['last_review'].fillna(data['last_review'].mode()[0])


# In[24]:


data['reviews_per_month'].value_counts()


# In[25]:


data['reviews_per_month'].mode()


# In[26]:


# impute with mode of reviews_per_month

data['reviews_per_month'] = data['reviews_per_month'].fillna(data['reviews_per_month'].mode()[0])


# In[27]:


data['host_name'].value_counts()


# In[28]:


data['host_name'].mode()


# In[29]:


# impute with mode of host_name

data['host_name'] = data['host_name'].fillna(data['host_name'].mode()[0])


# In[30]:


data.isnull().sum()


# # EDA

# In[31]:


# Target Variable

plt.figure(figsize=(12, 10))
sns.distplot(data['price'])
plt.tight_layout() 


# In[32]:


plt.figure(figsize=(12, 10))
sns.boxplot(data['price'])
plt.tight_layout() 


# In[33]:


data['price'].describe()


# In[34]:


sns.pairplot(data,diag_kind="hist")
plt.tight_layout() 


# In[35]:


data.hist(figsize= (16,9))


# In[36]:


data['room_type'].value_counts()


# In[37]:


plt.figure(figsize=(10, 6))
sns.countplot(x= 'room_type', order= data['room_type'].value_counts().index,data= data,palette='Accent')


# In[38]:


plt.figure(figsize=(15, 10))
sns.countplot(x= 'neighbourhood', order= data['neighbourhood'].value_counts().index,data= data,palette='Accent')

plt.xticks(
    rotation=45, 
    horizontalalignment='right'  
)

plt.tight_layout()


# In[39]:


plt.figure(figsize=(20, 10))

Filter_price = 800      # as the price is less than 1000 for the most of the houses

sub_data = data[data['price'] < Filter_price]

sns.violinplot(x= 'neighbourhood', y= 'price',data= sub_data,palette='Accent')

plt.xticks(
    rotation=45, 
    horizontalalignment='right'  
)

plt.tight_layout()


# In[40]:


plt.figure(figsize=(15,10))
sns.scatterplot(x="longitude", y="latitude", hue="room_type", data=data)                                                                                
plt.tight_layout()


# In[41]:


plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(), annot = True)
plt.tight_layout()


# # Converting Categorical Features

# In[42]:


data.dtypes


# In[43]:


data.nunique()


# In[44]:


data.head(10)


# In[45]:


data.drop(['id','name', 'host_name', 'last_review'],axis=1, inplace=True)


# In[ ]:


#data['last_review'].value_counts()
'''import datetime

data['last_review'] = pd.to_datetime(data['last_review'])'''
"import datetime\n\ndata['last_review'] = pd.to_datetime(data['last_review'])"


# In[46]:


data.head(10)


# In[47]:


data = pd.concat([data, pd.get_dummies(data['room_type'], prefix='d')],axis=1)
data.head()


# In[48]:


#convert to category dtype

data['neighbourhood'] = data['neighbourhood'].astype('category')
data.dtypes


# In[49]:


#use .cat.codes to create new colums with encoded value

data['neighbourhood'] = data['neighbourhood'].cat.codes
data.head()


# In[50]:


X = data.drop(['room_type', 'price'],axis=True)

y = data['price']


# In[51]:


X.head()


# In[52]:


y


# # Train - Test Split

# In[53]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[54]:


print('X_train:,y_train:',X_train.shape,y_train.shape)

print('*'*50)

print('X_test:,y_test:',X_test.shape,y_test.shape)


# # Model Building

# In[55]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

y_pred = lm.predict(X_test)

y_pred


# In[56]:


# import metrics library
from sklearn import metrics

# print result of MAE
print('MAE:  ',metrics.mean_absolute_error(y_test, y_pred))

# print result of MSE
print('MSE:  ',metrics.mean_squared_error(y_test, y_pred))

# print result of RMSE
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# print result of R^2 Score
print('R^2:  ',metrics.r2_score(y_test, y_pred))


# In[57]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=0)
dt.fit(X_train,y_train)

y_dt = dt.predict(X_test)
y_dt


# In[58]:


# import metrics library
from sklearn import metrics

# print result of MAE
print('MAE:  ',metrics.mean_absolute_error(y_test, y_dt))

# print result of MSE
print('MSE:  ',metrics.mean_squared_error(y_test, y_dt))

# print result of RMSE
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_dt)))

# print result of R^2 Score
print('R^2:  ',metrics.r2_score(y_test, y_dt))


# In[59]:


from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=0)
model_rf.fit(X_train,y_train)

y_rf = model_rf.predict(X_test)
y_rf


# In[60]:


# import metrics library
from sklearn import metrics

# print result of MAE
print('MAE:  ',metrics.mean_absolute_error(y_test, y_rf))

# print result of MSE
print('MSE:  ',metrics.mean_squared_error(y_test, y_rf))

# print result of RMSE
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_rf)))

# print result of R^2 Score
print('R^2:  ',metrics.r2_score(y_test, y_rf))


# In[61]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)

gb_pred = gbr.predict(X_test)
gb_pred


# In[62]:


# import metrics library
from sklearn import metrics

# print result of MAE
print('MAE:  ',metrics.mean_absolute_error(y_test, gb_pred))

# print result of MSE
print('MSE:  ',metrics.mean_squared_error(y_test, gb_pred))

# print result of RMSE
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, gb_pred)))

# print result of R^2 Score
print('R^2:  ',metrics.r2_score(y_test, gb_pred))

