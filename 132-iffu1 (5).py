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

# In[13]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[14]:


model1.summary()


# Performance metrics for model1

# In[15]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[16]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.tail()


# In[17]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["SP"]
df1.head()


# In[18]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["VOL"]
df1.head()


# In[19]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["WT"]
df1.head()


# In[20]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["HP"]
df1.head()


# In[21]:


cars =pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# In[22]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[23]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :",mse )
print("RMSE :",np.sqrt(mse))


# In[24]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[25]:


cars1 = cars.drop("WT", axis=1)
cars.head()


# In[26]:


model2=smf.ols("MPG~ HP+VOL+SP",data=cars1).fit()
model2.summary()


# In[27]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[28]:


pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[29]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# Observations from model2 summary()
# 
# The adjusted R-suared value improved slightly to 0.76
# 
# All the p-values for model parameters are less than 5% hence they are significant
# 
# therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG response variable
# 
# There is no improvement in MSE value

# In[30]:


cars1.shape


# #### Leverage (Hat Values):
# Leverage values diagnose if a data point has an extreme value in terms of the independent variables. A point with high leverage has a great ability to influence the regression line. The threshold for considering a point as having high leverage is typically set at 3(k+1)/n, where k is the number of predictors and n is the sample size.

# In[31]:


k = 3
n = 81
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[32]:


from statsmodels.graphics.regressionplots import influence_plot

influence_plot(model1,alpha=.05)

y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')

plt.show()


# Observations
# 
# From the av=bive plot,it is evident that data points 65,70,76,78,79,80 are the influencers
# 
# as their H Leverage values are higher and size is higher

# In[33]:


cars1[cars1.index.isin([65,70, 76,78,79,80]) ]


# In[34]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[35]:


cars


# In[36]:


model3= smf.ols('MPG~VOL+SP+HP',data = cars2).fit()


# In[37]:


model3.summary()


# # Performance Metrics for model3

# In[38]:


df3= pd.DataFrame()
df3["actual_y3"] =cars2["MPG"]
df3.head()


# In[39]:


df3= pd.DataFrame()
df3["actual_y3"] =cars2["SP"]
df3.head()


# In[40]:


df3= pd.DataFrame()
df3["actual_y3"] =cars2["HP"]
df3.head()


# In[41]:


df3= pd.DataFrame()
df3["actual_y3"] =cars2["VOL"]
df3.head()


# In[42]:


pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[43]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# In[44]:


Univ1 = Univ.iloc[:,1:]
Univ1


# In[ ]:


cols = Univ1.columns


# In[45]:


from sklearn.preprocessing import StandardSvcaler
scaler = StandardScaler()


# In[ ]:




