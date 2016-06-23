import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.linear_model.logistic import logisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score

df = pd.read_csv('SMSSpamCollection',delimiter='\t',header=None)

# print df
# print df.head()

print "Number of spam messages", df[df[0]=='spam'][0].count()
print "Number of ham messages", df[df[0]=='ham'][0].count()
