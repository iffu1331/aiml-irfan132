#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold


# In[3]:


dataframe = pd.read_csv("diabetes.csv")
dataframe


# In[9]:


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

X = dataframe.iloc[:,0:8]
Y = dataframe.iloc[:,8]

kfold = StratifiedKFold(n_splits=10,random_state = 3,shuffle = True)

model = RandomForestClassifier(n_estimators= 200,random_state= 20,max_depth=None)
results=cross_val_score(model, X, Y, cv=kfold)
print(results)
print(results.mean())


# In[15]:


val_counts()


# In[ ]:


from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(random)


# In[ ]:




