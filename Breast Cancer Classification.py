#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# # Importing breast cancer data set

# In[2]:


cancer_df = pd.read_csv('breast-cancer-wisconsin-data.csv')
cancer_df.shape


# In[3]:


cancer_df.head()


# In[4]:


cancer_df.tail()


# In[5]:


cancer_df.info()


# In[6]:


cancer_df.describe()


# In[7]:


clean_up_diagnosis = {"Diagnosis": {"M": 0,"B":1}}
cancer_df.replace(clean_up_diagnosis, inplace =True)
cancer_df.head()


# # Exploratory Data Analysis

# In[8]:


sns.pairplot(data = cancer_df, vars =['Mean Radius','Mean Texture', 'Mean Perimeter', 'Mean Area','Mean Smoothness','Mean Compactness'],hue='Diagnosis', palette = 'magma')


# In[9]:


sns.countplot(cancer_df['Diagnosis'],label = 'count', palette = 'magma')


# In[10]:


sns.distplot(cancer_df['Diagnosis'],color= 'orchid')


# In[11]:


sns.scatterplot( x= 'Mean Area',y= 'Mean Smoothness', data= cancer_df, hue ='Diagnosis',palette='magma')


# In[12]:


cancer_df_mean = cancer_df.ix[:,1:11]


# In[13]:


cancer_df_mean.plot(kind='density', subplots=True, layout=(4,3), figsize=(10,10), sharey=False, sharex=False, fontsize= 10)


# In[14]:


plt.figure(figsize=(22.5,12.5))
sns.heatmap(cancer_df.corr(),annot=True, lw=0.5)


# In[15]:


X = cancer_df.drop(['Diagnosis', 'ID Number'],axis=1)
X.shape


# In[16]:


X.head()


# In[17]:


X.tail()


# In[18]:


y = cancer_df['Diagnosis']
y.shape


# In[19]:


y.head()


# In[20]:


y.tail()


# # Train - Test Split and Model Training

# In[21]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

print(f' X train Shape: {X_train.shape}')
print(f' X test Shape: {X_test.shape}')
print(f' y train Shape: {y_train.shape}')
print(f' y test Shape: {y_test.shape}')


# In[22]:


svc= SVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)

cm= confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True,center=True, lw=0.5)

print(classification_report(y_test,y_pred))


# In[23]:


#Scaled values - Training set

min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train) /range_train

X_train_scaled


# In[24]:


sns.scatterplot(x=X_train['Mean Smoothness'], y=X_train['Mean Compactness'], data=X_train_scaled, hue = y_train, palette = 'magma')


# In[25]:


sns.scatterplot(x=X_train_scaled['Mean Smoothness'], y=X_train_scaled['Mean Compactness'], data=X_train_scaled, hue = y_train, palette = 'magma')


# In[26]:


#Scaled values - Test set

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# # Model Testing

# In[27]:


svc.fit(X_train_scaled,y_train)
y_pred = svc.predict(X_test_scaled)

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot = True, lw=0.5)

print(classification_report(y_test,y_pred))


# # Hyper - Tuning Parameters

# In[28]:


param_grid = {'C' : np.linspace(0.01,100,150), 'gamma': np.linspace(0.001,1,150), 'kernel': ['rbf','poly','linear'] }
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose=4)
grid.fit(X_train_scaled, y_train)


# In[29]:


print(f' Grid Search Best Parameters: \n\n {grid.best_params_} \n')
print(f' Grid Search Best Estimator: \n\n {grid.best_estimator_} \n')


# # Final Model and Predictions

# In[30]:


optimised_svm = grid.best_estimator_
optimised_svm.fit(X_train_scaled,y_train)

grid_predictions = optimised_svm.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True, lw=0.5)

print(classification_report(y_test, grid_predictions))


# In[31]:


results = pd.DataFrame(grid_predictions)
results.index = X_test_scaled.index
results.columns =['predictions']
results.sort_index().to_csv('Breast_Cancer_Prediction_Results.csv')


# In[ ]:




