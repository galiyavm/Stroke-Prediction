#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing Libraries


# In[346]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[347]:


df=pd.read_csv("/Users/mac/Desktop/health.csv") #importing dataset


# In[348]:


df


# In[349]:


df.head()#displaying first 5 rows


# In[350]:


df.tail()#displaying last 5 rows


# In[351]:


df.shape#gives the number of rows and columns


# In[352]:


df.info()#Showing information about dataset


# In[353]:


df.describe() #showing statistical features like mean,median,std dev etc of dataset.


# In[354]:


df.isnull().sum() #to find the null values


# In[355]:


sns.heatmap(df.isnull()) #to visualise the null values


# In[356]:


((df.isnull().sum())/df.shape[0])*100 # to find the percentage of null values


# In[357]:


df['bmi'].unique() #to get the unique bmi values


# In[358]:


df['bmi'].fillna(df['bmi'].mean(),inplace=True) #to fill the null values using fillna 


# In[359]:


df['bmi'].describe()


# In[360]:


df.isnull().sum()


# In[361]:


df.columns #to get the columns of dataset


# In[362]:


df.drop('id',axis=1,inplace=True)


# # Continuous variables

# In[363]:


df[['age','bmi','avg_glucose_level']] #these numerical columns are stored in a new variable


# In[366]:


df['age'].nunique()#gives unique values in this attribute


# In[365]:


df.age.value_counts()#count of each values


# In[342]:


sns.distplot(df.age)#The below code will plot a distribution of variable age


# In[367]:


df['bmi'].nunique()#returns the number of unique values


# In[302]:


df.bmi.value_counts()#This wil give the counts of each values


# In[304]:


sns.distplot(df.bmi)#The code helps to plot the distribution of variable bmi


# In[368]:


df['avg_glucose_level'].nunique()#returns number of unique values


# In[305]:


df.avg_glucose_level.value_counts()#gives the counts of the atrribute


# In[307]:


sns.distplot(df.avg_glucose_level) #to visualise how our variable average glucose level is distributed


# From these plots we get to know that all the numerical columns like avg glucose level,bmi,age are normally distributed.

# # Boxplot

# In[308]:


sns.boxplot(x=df['stroke'],y=df['age']) #boxplot between stroke and age
plt.show()


# People aged 60 years and more tend to have stroke.
# Some outliers can be seen as people below age 20 are having a stroke is also valid as stroke also 
# depends on our eating and living habits.

# In[309]:


sns.boxplot(x=df['stroke'],y=df['bmi'])#boxplot between stoke and bmi
plt.show

# There is no prominent observation of how does BMI affects the  chances of having the stroke.
# In[310]:


sns.boxplot(x=df['stroke'],y=df['avg_glucose_level'])#boxplot between stroke and avg glucose level
plt.show()


# People who experienced stroke were mostly among those who had high glucose level.Outliers are more in glucose level data.Stroke also happens more in the age of 60 to 80 years of age.
# 

# # Categorical columns

# In[311]:


df['gender'].value_counts()


# In[312]:


#visualising all the categorical columns using countplot


# In[314]:


df.gender.unique()


# In[315]:


sns.countplot(df['gender']) #visualising  using countplot


# In[316]:


df.ever_married.unique()


# In[317]:


df.ever_married.value_counts()


# In[318]:


sns.countplot(df['ever_married'])


# In[319]:


df.hypertension.value_counts()


# In[320]:


df.hypertension.unique()


# In[321]:


sns.countplot(df['hypertension'])


# In[322]:


df.heart_disease.value_counts()


# In[323]:


sns.countplot(df['heart_disease'])


# In[324]:


df.work_type.value_counts()


# In[325]:


df.work_type.unique()


# In[326]:


sns.countplot(df['work_type'])


# In[327]:


df.Residence_type.unique()


# In[328]:


df.Residence_type.value_counts()


# In[329]:


sns.countplot(df['Residence_type'])


# In[330]:


df.smoking_status.unique()


# In[331]:


df.smoking_status.value_counts()


# In[332]:


sns.countplot(df['smoking_status'])

#Visualising the categorical columns with our target variable - 'stroke'
# In[ ]:


# This plot helps us to analyze how gender will affect chances of stroke.


# In[336]:


sns.countplot(x=df['gender'],hue=df['stroke'])


# There is not much difference between stroke rate concerning gender.

#  This plot shows whether the person is ever married with respect to stroke

# In[337]:


sns.countplot(x=df['ever_married'],hue=df['stroke'])


# People who are married have higher rate of stroke.

# Below code will create a count plot of worktype with respect to stroke

# In[338]:


sns.countplot(x=df['work_type'],hue=df['stroke'])


# People working in the Private sector have a higher risk of getting a stroke. 
# And people who have never worked have a very less stroke rate.
# 

# The below code will create a counter plot
# Residence Type with respect to stroke

# In[339]:


sns.countplot(x=df['Residence_type'],hue=df['stroke'])


# There is not much difference in both attribute values.
# 
# 

# The below code will create a counter plot smoking status with respect to stroke

# In[340]:


sns.countplot(x=df['smoking_status'],hue=df['stroke'])


# As per these plots, we can see there is not much difference in the chances of stroke irrespective of smoking status.

# In[170]:


df.corr()


# In[281]:


sns.heatmap(df.corr(),annot=True)


# Variables that are showing some effective correlation are:
# 
# age, hypertension, heart_disease, avg_glucose level.

# # Label Encoding

# Our dataset is a mix of both categorical and numeric data 
# and since ML algorithms understand data of numeric nature let’s encode our categorical data into numeric ones using Label Encoder. Label Encoder is a 
# technique that will convert categorical data into numeric data.

# In[62]:


import sklearn
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[63]:


for i in range(0,df.shape[1]):
    if df.dtypes[i]=='object':
        df[df.columns[i]] = le.fit_transform(df[df.columns[i]])


# In[64]:


df.head()


# In[65]:


df.info()


# # Splitting the data for train and test

# In[66]:


# split df in to x and y
#putting feature variable to x
x=df.drop(['stroke'],axis=1)
#putting response variable to y
y=df['stroke']             


# In[67]:


x.head()


# In[68]:


#now lets split the data into train and test
from sklearn.model_selection import train_test_split


# In[69]:


#divide the data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[70]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Normalize

# In[71]:


#feature scaling

from sklearn.preprocessing import StandardScaler


# In[72]:


sc=StandardScaler()
x=sc.fit_transform(x)


# # Decision Tree

# In[92]:


# Decision tree

import sklearn
from sklearn.tree import DecisionTreeClassifier


# In[102]:


dc=DecisionTreeClassifier()
x=df.drop('stroke',axis=1)
y=df.stroke


# In[103]:


dc.fit(x_train,y_train)


# In[104]:


pred=dc.predict(x_test)


# In[105]:


pred[0:5]


# In[97]:


y_test[0:5]


# In[106]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[107]:


print(confusion_matrix(pred,y_test))


# In[108]:


print(classification_report(pred,y_test))


# In[110]:


accuracy_score(pred,y_test)


# # Random Forest

# In[111]:


rf = RandomForestClassifier(n_estimators=150,criterion='entropy') #define the model


# In[112]:


rf.fit(x_train,y_train) #fitting the model


# In[113]:


pred=rf.predict(x_test)


# In[114]:


pred[0:5]


# In[115]:


y_test[0:5]


# In[93]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[117]:


print(confusion_matrix(pred,y_test))


# In[118]:


print(classification_report(pred,y_test))


# In[119]:


accuracy_score(pred,y_test)


# # Logistic Regression

# In[111]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[115]:


lr.fit(x_train,y_train)


# In[117]:


y_pred=lr.predict(x_test)
ac_lr=accuracy_score(y_test,y_pred)
ac_lr


# # Conclusion

# In this mini-project, we saw some of the factors that might result in strokes. Age was highly correlated followed by hypertension, heart disease, avg glucose level, and ever married.
# 
# RandomForest performed very well.There are outliers in some variable, reason behind why I kept it as it is because these things are either depends on other factors and there are possibilities of having such kind of records.

# In[ ]:





# In[ ]:




