import numpy as np
from sklearn.datasets import load_boston
from scipy.stats import pointbiserialr, spearmanr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.metrics import accuracy_score,classification_report

#pointbiserialr for bnary classification and spearmanr for milti class classification
data = load_boston()

def scatter_plot(X,Y):
    plt.subplot(5,3,1)
    columns = ['crim','zn','indus','chos','nox','rm','age','dis','rad','tax','ptratio','b','lstat']

    subplot_counter = 1
    for c in range(len(columns)):
        x = X[:,c]
        y = Y
        plt.subplot(5,3,subplot_counter)
        plt.scatter(x,y)
        plt.axis("tight")
        # plt.title('Feature Selection', fontsize=14)
        plt.xlabel(columns[c], fontsize=12)
        plt.ylabel("price", fontsize=12)
        subplot_counter+=1
    plt.show()

def feature_selection(X,Y):
    param_df= ['crim','zn','indus','chos','nox','rm','age','dis','rad','tax','ptratio','b','lstat']
    # print param_df
    scores = []
    scoreCV =[]
    for j in range(5):
        scores = []
        scoreCV=[]
        for i in range(0,len(param_df)):
            # print df[:,0:i+1:]
            x = X[:,0:(i+1)]
            # print X
            y = Y
            clf =SGDRegressor(loss="squared_loss")
            scoreCV = cross_val_score(clf, x, y, cv=3)
            # print scoreCV
            # print np.mean(scoreCV)
            scores.append(np.mean(scoreCV))

        plt.figure(figsize=(15,5))
        plt.plot(range(1,len(scores)+1),scores, '.-')
        plt.axis("tight")
        plt.title('Feature Selection', fontsize=14)
        plt.xlabel('# Features', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.show()




def correlation(X,Y):
    param = []
    correlation=[]
    abs_corr=[]

    columns = ['crim','zn','indus','chos','nox','rm','age','dis','rad','tax','ptratio','b','lstat']

    for c in range(len(columns)):
        corr = spearmanr(X[:,c],Y)[0]
        correlation.append(corr)
        param.append(c)
        abs_corr.append(abs(corr))
    print correlation
    #create data frame
    param_cor = pd.DataFrame({'correlation':correlation,'parameter':param, 'abs_corr':abs_corr})
    paramc_cor=param_cor.sort_values(by=['abs_corr'], ascending=False)
    param_cor=param_cor.set_index('parameter')


    print param_cor





if __name__ == '__main__':
    data = load_boston()

    X = data.data
    Y = data.target



    X_train,X_test,Y_train,Y_test =  train_test_split(X,Y)

    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train)
    Y_train = Y_scaler.fit_transform(Y_train)
    X_test = X_scaler.transform(X_test)
    Y_test = Y_scaler.transform(Y_test)

    print X_train[0:5]


    print len(X_train)
    print Y_test

    clf =SGDRegressor(loss="squared_loss")
    scores = cross_val_score(clf,X_train,Y_train,cv=5)
    print scores
    print np.mean(scores)

    clf.fit_transform(X_train,Y_train)

    pred  = clf.predict(X_test)

    print  clf.score(X_test,Y_test)




    # correlation(X_train,Y_train)
    # feature_selection(X_train,Y_train)
    scatter_plot(X_train,Y_train)
