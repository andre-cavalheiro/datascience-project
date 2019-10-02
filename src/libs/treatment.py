import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif

def fixDataSet(df, label='class'):
    '''
        Transform string 'na' into NAN
        Transform dataset label into a boolean
    '''
    df = df.replace('na', np.nan)   # There's a way to do this right in read_csv
    print('Transforming label into boolean')
    '''
    df = df.replace({label: {       # Specific to this dataset!
        'false': False,
        'true': True
    }})'''
    df[label] = df[label].astype(bool)

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
    layers2drop = list()
    layers2keep = list()

    for item in topc.iteritems():
        col1 = item[0][0]
        col2 = item[0][1]
        corr_val = item[1]
        if corr_val > max_corr and col2 not in layers2keep and col1 not in layers2drop and col2 not in layers2drop:
            layers2drop.append(col2)
            layers2keep.append(col1)

    print('Dropping {} columns. New x state: {}'.format(len(layers2drop), df.shape))
    # print(' Columns: ', layers2drop)
    # print(df.shape)

    df = df.drop(layers2drop, axis=1)

    return df

def getTopCorrelations(df, n=None):
    '''
    Returns the 'n' most correlated labels

    Attributes:   df - the pandas dataframe with the dataset
                   n - the number of top instances to return
    Returns:      a Series with the 'n' most correlated labels
    '''
    corr_matrix = df.corr().abs()
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                 .stack()).sort_values(ascending=False)
    if n:
    	return sol[0:n]
    else:
    	return sol
