import numpy as np
from sklearn.datasets import load_boston
from scipy.stats import pointbiserialr, spearmanr
import pandas as pd
import matplotlib.pyplot as plt 

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
    correlation(X,Y)
    scatter_plot(X,Y)
