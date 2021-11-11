#!/usr/bin/env python
# coding: utf-8

# # importing the necessary dependencies

# In[1]:


import pandas as pd
import numpy as np
import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#print(stopwords.words('english'))


# # Data pre - processing

# In[3]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/fake news prediction/train.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


#checking for any null values
df.isnull().sum()


# In[7]:


#because of the size of dataset, we are replacing null values with empty strings
df = df.fillna('')


# In[8]:


df.isnull().sum()


# In[9]:


#creating a new column using author and title column
df['content'] = df['author'] + ' ' + df['title']


# In[10]:


df.head()


# Stemming
# 
# Stemming is a process which reduces a word to its root form

# In[11]:


#the stemming process
port_stem = PorterStemmer()


# In[12]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[13]:


df['content'] = df['content'].apply(stemming)


# In[14]:


print(df['content'])


# In[15]:


#separating the data and lables
X = df['content'].values
Y = df['label'].values


# In[16]:


print(X.shape)
print(Y.shape)


# In[17]:


#converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[18]:


print(X)


# # Splitting the data into training data and tesing data

# In[19]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, stratify = Y, random_state =2)


# In[20]:


#training the model
model = LogisticRegression()


# In[21]:


model.fit(x_train, y_train)


# In[22]:


#evaluation using accuracy score on tarining data
training_data_pred = model.predict(x_train)
accuracy_training_data = accuracy_score(y_train, training_data_pred)

print('ACCURACY OF TRAINING DATA IS :', accuracy_training_data)


# In[23]:


#evaluation using accuracy score on testing data
testing_data_pred = model.predict(x_test)
accuracy_testing_data = accuracy_score(y_test, testing_data_pred)

print('ACCURACY OF TRAINING DATA IS :', accuracy_testing_data)


# # making a predictive system

# In[25]:


a = int(input('TYPE THE INDEX OF THE DATA : '))
x_new = x_test[a]

prediction = model.predict(x_new)
#print(prediction)
print('THE PREDICTION VALUE IS :', prediction)
if prediction == 1:
    print('FAKE NEWS')
else:
    print('VALID NEWS')
    
print('THE ACTUAL VALUE IS :', y_test[a])


# In[ ]:




