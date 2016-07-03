import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

def load():
    df = pd.read_csv("seed.csv",delimiter=",",names=["area","perimeter","compactness","length","width","asymm","kernel","seed"])
    print df.head()
    print df.describe()
    return df

def scatter_plot(df):
    plt.subplot(4,2,1)
    columns = df.columns.values

    subplot_counter = 1
    for c in  columns:
        x = df[c]
        y = df["seed"]
        plt.subplot(4,2,subplot_counter)
        plt.scatter(x,y)
        plt.axis("tight")
        # plt.title('Feature Selection', fontsize=14)
        plt.xlabel(c, fontsize=12)
        plt.ylabel("seed", fontsize=12)
        subplot_counter+=1
    plt.show()

def feature_selection(df):
    param_df=df.columns.values
    print param_df
    scores = []
    scoreCV =[]
    for j in range(5):
        scores = []
        scoreCV=[]
        for i in range(0,len(param_df)-1):
            # print df[:,0:i+1:]
            X = df.ix[:,0:(i+1)]
            # print X
            y = df["seed"]
            clf = LogisticRegression()
            scoreCV = cross_validation.cross_val_score(clf, X, y, cv=3)

            print np.mean(scoreCV)
            scores.append(np.mean(scoreCV))

        plt.figure(figsize=(15,5))
        plt.plot(range(1,len(scores)+1),scores, '.-')
        plt.axis("tight")
        plt.title('Feature Selection', fontsize=14)
        plt.xlabel('# Features', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.show()



def correlation(df):
    columns = df.columns.values
    print columns
    param = []
    correlation=[]
    abs_corr=[]

    for c in columns:
        corr = spearmanr(df['seed'],df[c])[0]
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
    df = load()
    correlation(df)
    # scatter_plot(df)
    feature_selection(df)
