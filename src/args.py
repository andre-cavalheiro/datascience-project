from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier

from src.libs.balancing import *    # fixme !!! should not have src
from src.libs.treatment import *    # fixme !!! should not have src
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans, DBSCAN, SpectralClustering, AffinityPropagation, Birch,MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import f_classif, chi2, f_regression

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
            ('randomForest', RandomForestClassifier),
            ('ensembleBagging', BaggingClassifier),
            ('ensembleVoting', VotingClassifier)
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
        'name': 'correlationThreshold',
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
    {
        'name': 'clusterFunction',
        'type': str,
        'default': None,
        'help': 'Cluster function to be used',
        'required': False,
        'possibilities': [
            ('kmeans', KMeans),
            ('agglomerativeClustering', AgglomerativeClustering),
            ('miniBatchKMeans', MiniBatchKMeans),
            ('dbscan', DBSCAN),
            ('spectralClustering', SpectralClustering),
            ('affinityPropagation', AffinityPropagation),
            ('birch', Birch),
            ('GaussianMixture', GaussianMixture),
            ('meanShift', MeanShift),
        ]
    },
    {
        'name': 'featureFunction',
        'type': str,
        'default': None,
        'help': 'Feature selection fuction',
        'required': False,
        'possibilities': [
            ('f_classif', f_classif),
            ('chi2', chi2),
            ('f_regression', f_regression),
        ]
    },
    {
        'name': 'nFeatures',
        'type': int,
        'default': 10,
        'help': 'N features to select',
        'required': False,
    },
]

