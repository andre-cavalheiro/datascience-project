import pandas as pd
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
import time

def notBalancing(x, y):
    return x, y

def balancing(x, y, aloorithm, label='class', numSampPerClass=None):
    numSamplesPerClassType = {1: 10000, 2: 10000, 3: 10000, 4: 10000, 5: 10000, 6: 10000,
                              7: 10000}  # fixme - hard coded

    print('- Balancing with SMOTE')
    print('Current x state: ', x.shape)

    x_columns = x.columns.values
    counts = y.value_counts().to_dict()
    printStr = '> Initial class freq:\n'
    for k, v in counts.items():
        printStr += '"{}" instances: [{}]\n'.format(k, v)
    print(printStr)

    if numSampPerClass is not None:
        classNumSamp = {k: numSampPerClass[k] if k in numSampPerClass.keys() else v for k, v in counts.keys()}

        sm = SMOTE(random_state=int(time.time()), sampling_strategy=classNumSamp)
        rus = RandomUnderSampler(random_state=int(time.time()), sampling_strategy=classNumSamp)
        x_bal, y_bal = sm.fit_sample(x, y)
        x_bal, y_bal = rus.fit_sample(x_bal, y_bal)
    else:
        alg = aloorithm(random_state=int(time.time()))
        x_bal, y_bal = alg.fit_sample(x, y)

    x_bal = pd.DataFrame(x_bal, columns=x_columns)
    y_bal = pd.DataFrame(y_bal, columns=[label])

    # fixme - done in a stupid way
    df = x_bal.join(y_bal)
    y_bal = df.loc[:, label]
    x_bal = df.drop(columns=[label])

    counts = y_bal.value_counts().to_dict()
    printStr = 'Balanced class freq:\n'
    for k, v in counts.items():
        printStr += '"{}" instances: [{}]\n'.format(k, v)
    print(printStr)

    print('Balanced x state: ', x_bal.shape)

    return x_bal, y_bal


def smote(x, y, label='class', numSamplesPerClassType=None):
    # numSamplesPerClassType = {1: 10000, 2: 10000, 3: 10000, 4: 10000, 5: 10000, 6: 10000, 7: 10000}   # fixme - hard coded

    print('- Balancing with SMOTE')
    print('Current x state: ', x.shape)

    x_columns = x.columns.values
    counts = y.value_counts().to_dict()
    printStr = '> Initial class freq:\n'
    for k, v in counts.items():
        printStr += '"{}" instances: [{}]\n'.format(k, v)
    print(printStr)

    if numSamplesPerClassType is not None:
        classNumSamp = {k:v if v>counts[k] else counts[k] for k,v in numSamplesPerClassType.items()}
        sm = SMOTE(random_state=int(time.time()), sampling_strategy=classNumSamp)
    else:
        sm = SMOTE(random_state=int(time.time()))

    x_bal, y_bal = sm.fit_sample(x, y)

    x_bal = pd.DataFrame(x_bal, columns=x_columns)
    y_bal = pd.DataFrame(y_bal, columns=[label])

    # fixme - done in a stupid way
    df = x_bal.join(y_bal)
    y_bal = df.loc[:, label]
    x_bal = df.drop(columns=[label])

    counts = y_bal.value_counts().to_dict()
    printStr = 'Balanced class freq:\n'
    for k, v in counts.items():
        printStr += '"{}" instances: [{}]\n'.format(k, v)
    print(printStr)

    print('Balanced x state: ', x_bal.shape)

    return x_bal, y_bal


def randomUnderSample(x, y, label='class', numSamplesPerClassType=None):
    numSamplesPerClassType = {1: 100000, 2: 100000, 3: 100000, 4: 100000, 5: 100000, 6: 100000, 7: 100000}   # fixme - hard coded
    print('- Balancing with random under sampling')
    print('Current x state: ', x.shape)

    x_columns = x.columns.values
    counts = y.value_counts().to_dict()
    printStr = '> Initial class freq:\n'
    for k, v in counts.items():
        printStr += '"{}" instances: [{}]\n'.format(k, v)
    print(printStr)

    if numSamplesPerClassType is not None:
        classNumSamp = {k:v if v<counts[k] else counts[k] for k,v in numSamplesPerClassType.items()}
        rus = RandomUnderSampler(random_state=int(time.time()), sampling_strategy=classNumSamp)
    else:
        rus = RandomUnderSampler(random_state=int(time.time()))

    x, y = rus.fit_resample(x, y)

    x_bal = pd.DataFrame(x, columns=x_columns)
    y_bal = pd.DataFrame(y, columns=[label])

    # fixme - working but done in a stupid way
    df = x_bal.join(y_bal)
    y_bal = df.loc[:, label]
    x_bal = df.drop(columns=[label])

    counts = y_bal.value_counts().to_dict()
    printStr = 'Balanced class freq:\n'
    for k, v in counts.items():
        printStr += '"{}" instances: [{}]\n'.format(k, v)
    print(printStr)

    print('Balanced x state: ', x_bal.shape)

    return x_bal, y_bal


# Under-samples by replacing the original samples by the centroids of the cluster found.
def clusterCentroidsUnderSample(x, y, label='class'):
    print('Balancing with  clusterCentroids')
    print('Current x state: ', x.shape)

    x_columns = x.columns.values
    sampler = ClusterCentroids(random_state=0)

    x, y = sampler.fit_resample(x, y)
    print('Resampled dataset shape %s' % Counter(y))

    x_bal = pd.DataFrame(x, columns=x_columns)
    y_bal = pd.DataFrame(y, columns=[label])

    return x_bal, y_bal

