import arff #from liac-arff package
import pandas as pd, numpy as np
from IPython.display import display, HTML
from sklearn.preprocessing import LabelBinarizer #for dummification
from mlxtend.frequent_patterns import apriori, association_rules #for ARM
from sklearn.feature_selection import SelectKBest, f_classif
from treatment import discretize, dummify


#datapath = '../data/data/proj/pd_speech_features.csv'
#speechdata = pd.read_csv(datapath, header=1 )
#df = speechdata 
#print(df.head())


#Class notes
# Cross validation can't be done with the python package, because we need to preprocess the training data only after taking the test set
# 
# Stratified cross validation
# lift is better than confidence to assess pattern mining
#xgboost melhor que bagging do sklearn
#


def get_frequent_itemsets(dummified_df, minsup = 0.35, iteratively_decreasing_support = False, minpatterns = 30):
    frequent_itemsets = {}
    if iteratively_decreasing_support:
        while minsup > 0:
            minsup = minsup*0.9
            frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpatterns:
            print("Minimum support:",minsup)
            break
    else:
        frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
    print("Number of found patterns:",len(frequent_itemsets))
    return frequent_itemsets

#metric can also be lift
def get_association_rules(frequent_itemsets, metric="confidence", min_conf = 0.7, min_lift=1.2):
    minconf = 0.7
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    print(rules[(rules['antecedent_len']>=2)])
    return rules
