#!/usr/bin/env python
# coding: utf-8

# In[10]:


#importing necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#loading data_file
balance_data=pd.read_csv("F:\DataSets\Decision_Tree_ Dataset.csv",sep=',',header=0)


# In[6]:


print("Dataset Length: ",len(balance_data))
print("Dataset Shape: ", balance_data.shape)


# In[11]:


print("Dataset:: ")
print(balance_data.head())


# In[13]:


#separating target values
X=balance_data.values[:,0:4]
Y=balance_data.values[:,5]
#separating dataset into test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
#function to perform training with entrophy
clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)
clf_entropy.fit(X_train,Y_train)


# In[14]:


#function to make predictions
y_pred_en=clf_entropy.predict(X_test)
print(y_pred_en)


# In[15]:


#Checking Accuracy
print("Accuracy is ", accuracy_score(Y_test,y_pred_en)*100)


# In[ ]:




