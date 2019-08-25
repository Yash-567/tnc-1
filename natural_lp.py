# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 21:07:08 2019

@author: Yash Sonar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing tsv files
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
# quoting =3 is responsible for ignoring "


#Cleaning dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')
stopwords_list.remove('no')
stopwords_list.remove('not')
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    # ^ indicated not, so we remove everything except mentioned 
    review = review.lower()
    review = review.split() # converts string to list of words so we can access them by indexes
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords_list)]
    # we have used set() on stopwords to make it faster in case of larger documents
    review = ' '.join(review)
    corpus.append(review)
    
    
    
#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values


#splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.1, random_state=0)


# fitting naive bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)


# predicting the test result
Y_pred = classifier.predict(X_test)


# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)







