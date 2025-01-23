#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv("universities.csv")
df


# In[5]:


df.sort_values(by="GradRate",ascending=True)


# In[6]:


df.sort_values(by="GradRate",ascending=False)


# In[7]:


df[df["GradRate"]>=95]


# In[8]:


df[df["SAT"]>=95]


# In[9]:


df[(df["GradRate"]>=80) & (df["SFRatio"]<=12)]


# In[10]:


sal = pd.read_csv("Salaries.csv")
sal


# In[11]:


sal[["salary"]].groupby(sal["rank"]).mean()


# In[12]:


sal[["salary","phd","service"]].groupby(sal["rank"]).mean()


# In[13]:


np.mean(df["SAT"])


# In[14]:


np.median(df["SAT"])


# In[15]:


np.var(df["SFRatio"])


# In[16]:


df.describe()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


plt.figure(figsize=(6,3))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[23]:


s = [20,15,10,25,30,35,28,40,45,60]
scores = pd.Series(s)
scores


# In[24]:


plt.boxplot(scores, vert=False)


# In[26]:


s = [20,15,10,25,30,35,28,40,45,60,120,150]
scores = pd.Series(s)
scores


# In[27]:


plt.boxplot(scores, vert=False)


# In[28]:


s = [20,15,10,25,30,35,28,40,45,60,120,150]
scores = pd.Series(s)
scores


# In[29]:


plt.boxplot(scores, vert=True)


# In[30]:


df = pd.read_csv("universities.csv")
df


# In[31]:


s = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
scores = pd.Series(s)
scores


# In[39]:


plt.figure(figsize=(6,2))
plt.title("BOx plot for SAT score")
plt.boxplot(df["SAT"], vert = False)


# In[ ]:




