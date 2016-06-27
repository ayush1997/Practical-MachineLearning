import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('winequality-red.csv' ,sep=";")
print df
#gives all the  statistics
# print df.describe()

# print df["quality"]
# print df.columns[:-1]


# plt.scatter(df["alcohol"],df["quality"])
# plt.xlabel("Alcohol Content")
# plt.ylabel("Quality")
# plt.title("wijen quality analy")
# plt.show()



X = df[list(df.columns)[:-1]]
Y = df["quality"]
# print X
# print Y

X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
print X_train
print Y_train

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

scores = cross_val_score(regressor,X,Y,cv = 5)
print scores.mean(),scores

y_prediction = regressor.predict(X_test)

# print "prediction = ",y_prediction

print "r squared=",regressor.score(X_test,Y_test)
