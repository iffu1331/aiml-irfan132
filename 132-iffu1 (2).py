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

# In[3]:


cars.info()


# In[4]:


cars.isna().sum()


# Observations about info(), missing values
# 
# There are no missing values
# 
# There are 81 observations(81 different cars data)
# The data types of the columns are also relevant and valid

# In[5]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[6]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[7]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[8]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[9]:


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

# In[10]:


cars[cars.duplicated()]


# Pair plots and Correlation Coefficients

# In[11]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[12]:


cars.corr()


# Observations from correlation plots and Coefficients
# 
# Between x and y, all the x variables are showing moderate to high correlation strengths,highest being betwwen HP and MPG
# 
# Therefore this dataset qualifies for building a mutiple linear regression model to predict MPG
# 
# Among x columns (x1,x2,,x3 and x4),some very high correlation strenghts are observed between SP andHP,VOL vs WT
# 
# The high correlation among x columns is not desirable as it might lead to multicollinearity problem

# Preparing a preliminary model considering all X columns

# In[14]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[15]:


model1.summary()


# Performance metrics for model1

# In[18]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[19]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.tail()


# In[22]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["SP"]
df1.head()


# In[23]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["VOL"]
df1.head()


# In[24]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["WT"]
df1.head()


# In[25]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["HP"]
df1.head()


# In[26]:


cars =pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# In[27]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[ ]:




