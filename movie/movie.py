import pandas as pd

df =pd.read_csv("train.tsv",header=0,delimiter='\t')

# print df
# print df.describe()
# print df.count()
# print df.head()

print df['Phrase'].head(10)

print df['Sentiment'].describe()
print df['Sentiment'].value_counts()
print df['Sentiment'].count()

print df['Sentiment'].value_counts()/df['Sentiment'].count()
