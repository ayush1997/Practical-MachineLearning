import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score

df = pd.read_csv('SMSSpamCollection',delimiter='\t',header=None)

# print df
# print df.head()

print "Number of spam messages", df[df[0]=='spam'][0].count()
print "Number of ham messages", df[df[0]=='ham'][0].count()


print type(df[0])
X_train_raw,X_test_raw,Y_train,Y_test =  train_test_split(df[1],df[0])

print X_train_raw
print Y_train

vectorizer = TfidfVectorizer()
X_train =vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
