import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('SMSSpamCollection',delimiter='\t',header=None)

# print df
# print df.head()

print "Number of spam messages", df[df[0]=='spam'][0].count()
print "Number of ham messages", df[df[0]=='ham'][0].count()


# print type(df[0])
X_train_raw,X_test_raw,Y_train,Y_test =  train_test_split(df[1],df[0])

print X_test_raw
print Y_train

vectorizer = TfidfVectorizer()
X_train =vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

print X_train
print "test",X_test

classifier = LogisticRegression()
classifier.fit(X_train,Y_train)
predictions = classifier.predict(X_test)

X_test_raw = list(X_test_raw)

for i,prediction in enumerate(predictions[:5]):
    print "Prediction :",prediction,"messages:",X_test_raw[i]

print "accuracy_score:",accuracy_score(Y_test,predictions)


scores = cross_val_score(classifier,X_train,Y_train,cv=5)
print np.mean(scores),scores




# performance metrics

# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
#
# y_test = [0,0,0,0,0,1,1,1,1,1]
# y_pred = [0,1,0,0,0,0,0,1,1,1]
#
# confusion_matrix = confusion_matrix(y_test,y_pred)
# print confusion_matrix
#
# plt.matshow(confusion_matrix)
# plt.title('confusion matric')
# plt.colorbar()
#
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
#
# plt.show()
