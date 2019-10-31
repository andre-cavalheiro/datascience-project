import arff #from liac-arff package
import pandas as pd, numpy as np
from IPython.display import display, HTML
from sklearn.preprocessing import LabelBinarizer #for dummification
from mlxtend.frequent_patterns import apriori, association_rules #for ARM
from sklearn.feature_selection import SelectKBest, f_classif


#datapath = '../data/data/proj/pd_speech_features.csv'
#speechdata = pd.read_csv(datapath, header=1 )
#df = speechdata 
#print(df.head())


#Discretize real-valued attributes: width (pd.cut)
def discretize_width(df):
    newdf = df.copy()
    for col in newdf:
        if col not in ['class','id','gender']: 
            newdf[col] = pd.cut(newdf[col],3,labels=['0','1','2'])
    print(newdf.head(5))
    return newdf


#Discretize real-valued attributes: depth (pd.qcut)
def discretize_depth(df):
    newdf = df.copy()
    for col in newdf:
        if col not in ['class','id','gender']: 
            newdf[col] = pd.qcut(newdf[col],3,labels=['0','1','2'])
    print(newdf.head(5))
    return newdf

#Class notes
# Cross validation can't be done with the python package, because we need to preprocess the training data only after taking the test set
# 
# Stratified cross validation
# lift is better than confidence to assess pattern mining
#xgboost melhor que bagging do sklearn
#

# dummify
def dummify(newdf):
    dummylist = []
    for att in newdf:
        if att in ['class','id','gender']: newdf[att] = newdf[att].astype('category')
        dummylist.append(pd.get_dummies(newdf[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)
    print(dummified_df.head(5))
    return dummified_df

def get_frequent_itemsets(dummified_df):
    minsup = 0.35 #you can also use iteratively decreasing support as in the previous example
    frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
    print(frequent_itemsets)
    return frequent_itemsets

def get_association_rules(frequent_itemsets):
    minconf = 0.7
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    print(rules[(rules['antecedent_len']>=2)])
    return rules

"""
columns = SelectKBest(f_classif, k=10).fit(x, y).get_support()
new_x = x.loc[:,columns]
dummify(discretize(new_x))

#dw = discretize_width(df)
#dummies = dummify(dw)
#freqs = get_frequent_itemsets(dummies)
#get_association_rules(freqs)

patternMining(df)
"""