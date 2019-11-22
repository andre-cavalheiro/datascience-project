import pandas as pd, numpy as np
from IPython.display import display, HTML
from sklearn.preprocessing import LabelBinarizer #for dummification
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules #for ARM
from sklearn.feature_selection import SelectKBest, f_classif


#Class notes
# Cross validation can't be done with the python package, because we need to preprocess the training data only after taking the test set
# 
# Stratified cross validation
# lift is better than confidence to assess pattern mining
# xgboost melhor que bagging do sklearn
#

#apriori is slow, eclat or fp-tree is faster
def get_frequent_itemsets(dummified_df, minsup = 0.35, iteratively_decreasing_support = False, minpatterns = 30, alg = "fpgrowth"):
    frequent_itemsets = {}
    if alg == "fpgrowth":
        freq_func = fpgrowth
    else:
        freq_func = apriori
    if iteratively_decreasing_support:
        while minsup > 0:
            minsup = minsup*0.9
            frequent_itemsets = freq_func(dummified_df, min_support=minsup, use_colnames=True)
            
            if len(frequent_itemsets) >= minpatterns:
                print("Minimum support:",minsup)
                break
    else:
        frequent_itemsets = freq_func(dummified_df, min_support=minsup, use_colnames=True)
    print("Number of found patterns:",len(frequent_itemsets))
    return frequent_itemsets

#metric can also be lift
def get_association_rules(frequent_itemsets, metric="confidence", min_conf = 0.7, min_lift=1.2):
    minconf = 0.7
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    pd.set_option('display.max_columns', None)
    print(rules.sort_values('lift', ascending=False))
    return rules
