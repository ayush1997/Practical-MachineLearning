import numpy as np
from sklearn.datasets import load_boston
from scipy.stats import pointbiserialr, spearmanr
import pandas as pd

#pointbiserialr for bnary classification and spearmanr for milti class classification
data = load_boston()

# print data.data
# print len(data.target)

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
