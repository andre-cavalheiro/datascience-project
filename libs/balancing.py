import pandas as pd
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
import time

def notBalancing(x, y):
    return x, y

def smote(x, y, numMinorityIntances=False, label='class'):
    # label='consensus'
    x_columns = x.columns.values

    if numMinorityIntances:
        counts = y.value_counts().to_dict()
        ratio = {
            False: counts[False],
            True: numMinorityIntances
        }
        sm = SMOTE(random_state=420, sampling_strategy=ratio)
    else:
        sm = SMOTE(random_state=420)

    x_bal, y_bal = sm.fit_sample(x, y)

    x_bal = pd.DataFrame(x_bal, columns=x_columns)
    y_bal = pd.DataFrame(y_bal, columns=[label])

    # fixme - done in a stupid way
    df = x_bal.join(y_bal)
    y_bal = df.loc[:, label]
    x_bal = df.drop(columns=[label])

    print('Balanced with SMOTE. Current x state: ', x_bal.shape)
    return x_bal, y_bal

def randomUnderSample(x, y, numMajorityIntances=False, label='class'):
    print('Balancing with random under sampling')
    print('Current x state: ', x.shape)

    print(x.head(5))
    x_columns = x.columns.values
    if numMajorityIntances == False:
        rus = RandomUnderSampler(random_state=int(time.time()))
    else:
        counts = y.value_counts().to_dict()
        ratio = {
            False: numMajorityIntances,
            True: counts[True]
        }
        rus = RandomUnderSampler(random_state=int(time.time()), sampling_strategy=ratio)
    x, y = rus.fit_resample(x, y)

    print('Resampled dataset shape %s' % Counter(y))

    x = pd.DataFrame(x, columns=x_columns)
    y = pd.DataFrame(y, columns=[label])

    df = x.join(y)
    print('Balanced Dataframe: ', df.shape)
    print(df.shape)
    return df, 'undersample'


# Under-samples by replacing the original samples by the centroids of the cluster found.
def clusterCentroidsUnderSample(x, y, label='class'):
    print('Balancing with  clusterCentroids')
    print('Current x state: ', x.shape)
    print(x.head(5))
    x_columns = x.columns.values
    sampler = ClusterCentroids(random_state=0)
    x, y = sampler.fit_resample(x, y)

    print('Resampled dataset shape %s' % Counter(y))

    x = pd.DataFrame(x, columns=x_columns)
    y = pd.DataFrame(y, columns=[label])

    df = x.join(y)
    print('Balanced Dataframe: ', df.shape)
    print(df.shape)
    return df, 'clusterCentroids'

