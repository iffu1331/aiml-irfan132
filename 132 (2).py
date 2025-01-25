#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[6]:


data.info()


# In[7]:


print(type(data))
print(data.shape)
print(data.size)


# In[8]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[9]:


data1.info()


# In[10]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[11]:


data[data1.duplicated(keep = False)]


# In[12]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[13]:


data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# impute the missing values

# In[14]:


data1.info()


# In[16]:


data1.isnull().sum()


# In[23]:


cols = data1.columns
colors = ['white','green']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[18]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[19]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[25]:


data1['Solar'] = data1['Solar'].fillna(mean_ozone)
data1.isnull().sum()


# In[ ]:




