#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# Assumptions in multilinear regression
# 
# 
# 
# 1.Linearity: THe linearity between x and y is linear
# Independence: Observations are independent to each other
# 
# Homoscedasticity: The residuals (Y-Y_hat)exhibit constamt variance at alla levels of the predictor
# 
# Normal Distribution of Errors: The residuals of the model are normally distributed.
# 
# No multicollinearly: The independent variables should not be too highlt correlated with each other.
# 
# Violations of these assumptions may lead to inefficiency in the regression parametere and unreliable predictions.

# # EDA

# In[8]:


cars.info()


# In[9]:


cars.isna().sum()


# Observations about info(), missing values
# 
# There are no missing values
# 
# There are 81 observations(81 different cars data)
# The data types of the columns are also relevant and valid

# In[11]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[12]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[14]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[15]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[18]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# Observations from boxplot and histograms
# 
# There are some extreme values (outliers) observed in towards the right tail of SP and HP distributions
# 
# In VOL andWT columns, a few outliers are observed in both tails of their distributors.
# 
# The extreme values of cars data may have come from the specially designed nature of cars.
# 
# As this is multi-dimensional data,the outliers with respect to spatial dimensions may have to be cosidered while building the regression model

# In[21]:


cars[cars.duplicated()]


# Pair plots and Correlation Coefficients

# In[25]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[26]:


cars.corr()


# In[ ]:


Observations from Pairplot and correlation

