#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[15]:


data1.isnull().sum()


# In[19]:


cols=data1.columns
colors=['white','black']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar=True)


# In[20]:


data1.describe()


# In[21]:


data1.boxplot()


# In[22]:


fig, axes = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': [1,3]})

sns.histplot(data1["Newspaper"],kde=True, ax=axes[1], color='lightgreen',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_ylabel("Newspaper Levels")
axes[1].set_xlabel("Frequency")


# In[23]:


plt.scatter(data1["Newspaper"],data1["daily"])


# In[24]:


plt.scatter(data1["daily"],data1["sunday"])


# In[ ]:





# In[ ]:




