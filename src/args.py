from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from src.libs.balancing import *    # fixme !!! should not have src
from src.libs.treatment import *    # fixme !!! should not have src


argListPuppet = [
    {
        'name': 'dataset',
        'type': str,
        'default': None,
        'required': True,
        'help': 'Dataset to be used',

},
    {
        'name': 'classifier',
        'type': str,
        'default': None,
        'help': 'Classifier to be used.',
        'required': False,
        'possibilities': [
            ('KNN', KNeighborsClassifier),
            ('naiveBays', GaussianNB),
            ('decisionTree', DecisionTreeClassifier),
            ('randomForest', RandomForestClassifier)
        ]
    },
    {
        'name': 'classifierParams',
        'type': str,
        'default': None,
        'help': 'Possible parameters to be passed to the classifier.',
        'required': False,
        'children': [{              # Example params for knn
            'name': 'n_neighbors',
            'optimize': False,
            'optimizeInt': [1, 10]
        },
        {
            'name': 'weights',
        },
            {
            'name': 'metric',
        }
        ]
    },
    {
        'name': 'nanStrategy',
        # possibilities are median, most_frequent, ''  -> the possibilities key should not be added since this isn't a
        # dynamic argument
        'type': str,
        'default': '',
        'help': 'Nan strategy to be used.',
        'required': False,
        'optimize': False,
        'optimizeCategorical': ['median', 'most_frequent']
    },
    {
        'name': 'balancingStrategy',
        'type': str,
        'default': None,
        'help': 'Balancing strategy to be used.',
        'required': True,
        'possibilities': [
            ('smote', smote),
            ('randUndersample', randomUnderSample),
            ('clusterCentroidsUnderSample', clusterCentroidsUnderSample),
            ('notBalanced', notBalancing)],
        'optimize': False,
        'optimizeCategorical': [
            ('smote'),
            ('notBalanced'),
            ('randUndersample'),
            # ('clusterCentroidsUnderSample'),
        ]

    },
    {
        'name': 'covarianceThreshold',
        'type': float,
        'default': None,
        'help': 'Covariance threshold value to be used.',
        'required': True,
        'optimize': True,
        'optimizeUniform': [0.50, 0.99]
    },
{
        'name': 'rescaler',
        'type': str,
        'default': None,
        'help': 'Method of rescaling numbers',
        'required': True,
        'possibilities': [
            ('normalize', normalize),
            ('standardize', standardize),
            ('standardizeRobust', standardizeRobust)
        ],
    },
    {
        'name': 'percComponentsPCA',
        'type': int,
        'default': None,
        'help': 'Percentages of components to be used in the PCA process',
        'required': False,
        'optimize': False,
        'optimizeUniform': [0.30, 0.99]
    },
]

