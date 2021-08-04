# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 20:34:26 2021

@author: SHASHWAT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('https://raw.githubusercontent.com/krishnaik06/SpamClassifier/master/smsspamcollection/SMSSpamCollection',sep='\t',names=['label','msg'])

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
corpus=[]

for i in range(len(df)):
    review=re.sub('[^a-zA-Z]',' ',df['msg'][i])
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english') ]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)

x=cv.fit_transform(corpus).toarray()

import pickle
pickle.dump(cv,open('transform.pkl','wb'))

y=pd.get_dummies(df['label'],drop_first=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

pickle.dump(model,open('nlp_model.pkl','wb'))

