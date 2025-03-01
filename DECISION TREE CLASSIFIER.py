#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[8]:


iris1=pd.read_csv("iris1.csv")
print(iris1)


# In[10]:


import seaborn as sns
counts = iris1["variety"].value_counts()
sns.barplot(data = counts)


# In[11]:


iris1.info()


# In[12]:


iris1[iris1.duplicated(keep=False)]


# In[13]:


labelencoder = LabelEncoder()
iris1.iloc[:, -1] = labelencoder.fit_transform(iris1.iloc[:, -1])
iris1.head()


# In[14]:


iris1.info()


# In[15]:


iris1['variety'] = pd.to_numeric(labelencoder.fit_transform(iris1['variety']))
print(iris1.info())


# In[16]:


iris1.head(3)


# In[17]:


X=iris1.iloc[:,0:4]
Y=iris1['variety']


# In[24]:


x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3,random_state=1)
x_train


# In[25]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth =None)
model.fit(x_train,y_train)


# In[26]:


plt.figure(dpi=1200)
tree.plot_tree(model);


# In[ ]:




