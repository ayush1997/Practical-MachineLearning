
##################################################################################################################
#                                                                                                                #
#                UDER CONSTRUCTION                                                                               #
#                                                                                                                #
##################################################################################################################


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV





# print df
# print df.describe()
# print df.count()
# print df.head()

# print df['Phrase'].head(10)
#
# print df['Sentiment'].describe()
# print df['Sentiment'].value_counts()
# print df['Sentiment'].count()
#
# print df['Sentiment'].value_counts()/df['Sentiment'].count()

def main():
    pipeline = Pipeline([
    ('vect',TfidfVectorizer(stop_words='english')),
    ('clf',LogisticRegression())
    ])

    parameters = {
        'vect__max_df':(0.25,0.5),
        'vect__ngram_range':((1,1),(1,2)),
        'vect__use_idf':(True,False),
        'clf__C':(0.1,1,10),
    }

    df =pd.read_csv("train.tsv",header=0,delimiter='\t')
    X,y = df['Phrase'],df['Sentiment'].as_matrix()
    X_train,X_test,Y_train,Y_test =  train_test_split(X,y,train_size=0.5)
    grid_search =GridSearchCV(pipeline,parameters,n_jobs=3,verbose=1,scoring='accuracy')
    grid_search.fit(X_train,Y_train)
    print "Best score",grid_search.best_score_
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print param_name,best_parameters[param_name]

if __name__ == '__main__':
    main()
