#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy import stats
from sklearn import *


# 

# In[4]:


# Load Titanic dataset (can be downloaded from Kaggle or use this GitHub-hosted CSV)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display the first 5 rows of the dataset
df.head()


# In[5]:


# Check for missing values
missing_values = df.isnull().sum()
print("Missing values before imputation:\n", missing_values)

# Impute missing values:
# - 'Age' ➜ fill with median
# - 'Embarked' ➜ fill with mode (most frequent value)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop rows where target variable 'Survived' is missing (shouldn't be many)
df.dropna(subset=['Survived'], inplace=True)

# Check again after imputation
missing_values_after = df.isnull().sum()
print("\nMissing values after imputation:\n", missing_values_after)


# In[6]:


# Function to remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Apply IQR method to remove outliers in the 'Age' and 'Fare' columns
df = remove_outliers_iqr(df, 'Age')
df = remove_outliers_iqr(df, 'Fare')

# Display the shape of the dataset after removing outliers
print("\nDataset shape after removing outliers:", df.shape)


# In[7]:


# One-Hot Encoding for categorical variables 'Sex' and 'Embarked'
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Display the first few rows of the encoded dataset
df_encoded.head()


# In[8]:


# Plot histograms for numerical features
df_encoded[['Age', 'Fare']].hist(bins=20, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.show()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot count plot for 'Survived' (target variable)
plt.figure(figsize=(10, 6))
sns.countplot(data=df_encoded, x='Survived', palette='pastel')
plt.title('Count Plot for Survived', fontsize=16)
plt.show()

# Plot count plot for 'Embarked' (categorical feature)
plt.figure(figsize=(10, 6))
sns.countplot(data=df_encoded, x='Embarked_S', palette='pastel')
plt.title('Count Plot for Embarked', fontsize=16)
plt.show()

