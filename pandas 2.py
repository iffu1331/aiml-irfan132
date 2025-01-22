#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[5]:


df = pd.read_csv("universities.csv")
df


# In[6]:


df.sort_values(by="GradRate",ascending=True)


# In[7]:


df.sort_values(by="GradRate",ascending=False)


# In[8]:


df[df["GradRate"]>=95]


# In[9]:


df[df["SAT"]>=95]


# In[10]:


df[(df["GradRate"]>=80) & (df["SFRatio"]<=12)]


# In[11]:


sal = pd.read_csv("Salaries.csv")
sal


# In[13]:


sal[["salary"]].groupby(sal["rank"]).mean()


# In[16]:


sal[["salary","phd","service"]].groupby(sal["rank"]).mean()


# In[17]:


np.mean(df["SAT"])


# In[18]:


np.median(df["SAT"])


# In[19]:


np.var(df["SFRatio"])


# In[20]:


df.describe()


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[27]:


plt.figure(figsize=(6,3))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[ ]:




