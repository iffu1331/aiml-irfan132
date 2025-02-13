#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[3]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[4]:


titanic.info()


# In[6]:


titanic.columns


# Observations:
#     
# 1.Datatype is same for all the columns.
# 
# 2.There are no null values.
# 
# 3.As the columns are categorical,we can  adopt one-hot-encoding.

# In[13]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[19]:


counts = titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# In[20]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# In[21]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# Observations:
#     
# Crew are highest and second highest are 3rd,1st and 2nd are having in the third and fourth places.

# In[15]:


titanic['Age'].value_counts()


# In[17]:


df=pd.get_dummies(titanic,dtype=int)
df.head()


# In[22]:


df.info()


# In[33]:


frequent_itemsets.info()


# # Apriori Algorithm

# In[31]:


rules = association_rules(frequent_itemsets,metric="lift",min_threshold=1.0)
rules


# In[34]:


rules = association_rules(frequent_itemsets,metric="confidence",min_threshold=1.0)
rules


# In[35]:


rules.sort_values(by='lift', ascending = True)


# In[36]:


rules.sort_values(by='confidence', ascending = True)


# In[39]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




