import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.decomposition import PCA

def fixDataSetSpeach(df, label='class'):
    '''
        Transform string 'na' into NAN
        Transform dataset label into a boolean
    '''
    df = df.replace('na', np.nan)   # There's a way to do this right in read_csv
    print('Transforming label into boolean')
    df[label] = df[label].astype(bool)

    # print(df.head(5))
    return df

def fixDataSetCov(df, label='class'):
    '''
        Transform string 'na' into NAN
        Transform dataset label into a boolean
    '''
    df = df.replace('na', np.nan)   # There's a way to do this right in read_csv
    print('Transforming label into boolean')
    df = df.rename({df.shape[1] - 1: label}, axis='columns')

    # print(df.head(5))
    return df

def fillNan(df, strategy):

    '''
    obsWithMostNan = df.apply(lambda x: df.isnull().sum(), axis=1).head(10)
    print('Top 10 biggest percentage of NaN values per observation:')
    print(obsWithMostNan)
    '''
    if strategy == '':
        return df
    print('Number of missing values: {}. Filling NaN with {}'.format(df.isnull().sum().sum(), strategy))
    attrWithMostNan = df.isnull().sum().divide(df.shape[0]).sort_values(ascending=False).head(10)
    # print('Top 10 biggest percentage of NaN values per attributes: {}'.format(attrWithMostNan))
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputer = imputer.fit(df.iloc[:, 1:df.shape[1]])
    df.iloc[:, 1:df.shape[1]] = imputer.transform(df.iloc[:, 1:df.shape[1]])
    # print('Finished, number of missing values: ', df.isnull().sum().sum())
    # print(df.head(5))
    return df


def applyPCA(x, numComponentsToKeep):
    print('Applying PCA, keeping {} components'.format(numComponentsToKeep))
    columns = ['Component {}'.format(i) for i in range(numComponentsToKeep)]
    pca = PCA(n_components=numComponentsToKeep)

    principalComponents = pca.fit_transform(x)
    x_transf = pd.DataFrame(data=principalComponents,
                               columns=columns)
    return x_transf


def standardize(x):
    '''
     Standarlize data  (removing the mean and scaling to unit variance)
    '''
    print('Standardizing')
    columns = x.columns.values
    standard_scaler = StandardScaler()
    x_stand = standard_scaler.fit_transform(x)  # Returns numpy array
    x_stand = pd.DataFrame(x_stand, columns=columns)
    # print(x_stand.head(5))
    return x_stand

def normalize(x):
    print('Normalizing')
    columns = x.columns.values
    min_max_scaler = MinMaxScaler()
    x_norm = min_max_scaler.fit_transform(x)
    x_norm = pd.DataFrame(x_norm, columns=columns)
    # print(x_norm.head(5))
    return x_norm

def standardizeRobust(x):
    print('Standardizing robust')
    columns = x.columns.values
    robust_scaler = RobustScaler()
    x_stand = robust_scaler.fit_transform(x)
    x_stand = pd.DataFrame(x_stand, columns=columns)
    # print(x_norm.head(5))
    return x_stand

def dropHighCorrFeat(df, max_corr, n=None):
    '''
    Elimina as colunas do dataframe onde a correlaçao é maior que max_corr
    Attributes:	df - the pandas dataframe with the dataset
                    max_corr - the maximum acceptable value for the correlation
                    n - number of top correlations to consider. All if n=None
    '''
    print('Applying covariance threshold of {}, current x state {}'.format(max_corr, df.shape))

    topc = getTopCorrelations(df, n)

    variableCorr = [{'x': item[0][0], 'y': item[0][1], 'corr': item[1]} for item in topc.iteritems()]
    variableFrequencies = {}

    for v in variableCorr:
        if v['corr'] > max_corr:
            if v['x'] in variableFrequencies.keys():
                variableFrequencies[v['x']]+=1
            else:
                variableFrequencies[v['x']]=1
            if v['y'] in variableFrequencies.keys():
                variableFrequencies[v['y']]+=1
            else:
                variableFrequencies[v['y']]=1
    layers2drop = list()
    for v in variableCorr:
        if v['corr'] > max_corr:
            toDrop = v['x'] if variableFrequencies[v['x']] > variableFrequencies[v['y']] else v['y']
            if toDrop not in layers2drop:
                layers2drop.append(toDrop)

    df = df.drop(layers2drop, axis=1)

    print('Dropping {} columns. New x state: {}'.format(len(layers2drop), df.shape))
    # print(' Columns: ', layers2drop)
    # print(df.shape)

    return df, layers2drop

def getTopCorrelations(df, n=None):
    '''
    Returns the 'n' most correlated labels

    Attributes:   df - the pandas dataframe with the dataset
                   n - the number of top instances to return
    Returns:      a Series with the 'n' most correlated labels
    '''
    corr_matrix = df.corr(method='spearman').abs()
    uperCorrMatrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    sol = (uperCorrMatrix.stack())\
                .sort_values(ascending=False)
    if n:
    	return sol[0:n]
    else:
    	return sol
