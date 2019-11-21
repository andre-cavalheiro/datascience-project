from sys import exit
from libs.utils import *
from libs.dir import *
from libs.standardPlots import *
from src.puppet import Puppet
from src.args import argListPuppet
from argsConf.jarvisArgs import argListJarvis
from argsConf.plotArgs import argListPlots
from os.path import join

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier

# Verify command line arguments
parser = argparse.ArgumentParser(description='[==< J A R V I S - (Simplefied mooshak  version) >==]')

parser.add_argument('numLines', type=int, help='')
parser.add_argument('dataset', type=str, help='')
parser.add_argument('task', type=str, help='')

clArgs = parser.parse_args()
clArgs = vars(clArgs)      # Convert to dictionary
printDict(clArgs, statement="> JARVIS (simplified) using args:")

testDirectory = 'testConfigs(DoNotCHANGE)'
outputDir = join(testDirectory, 'output')
successString = ' - finished'


# Param verification and initialization
assert(type(clArgs['numLines']) == int)
assert(clArgs['numLines'] != 0)

if clArgs['dataset'] == 'PD':
    dataset = 'src/data/pd_speech_features.csv'
elif clArgs['dataset'] == 'CT':
    dataset = 'src/data/covtype.data'
else:
    print('Unknown dataset: "{}" - Only options are [PD] (pd_speech) or [CT] (covtype)'.format(clArgs['DATASET']))
    exit()

if clArgs['task'] == 'preprocessing':
    print('Unimplemented :c')
    exit()

elif clArgs['task'] == 'unsupervised':
    print('Unimplemented :c')
    exit()

elif clArgs['task'] == 'classification':
    conf = join(testDirectory, 'classification.yaml')
    changes = [{
        'classifier': DecisionTreeClassifier,
        'name': 'Decision Tree',
        'classifierParams': {
            'criterion': 'gini',
            'max_depth': 30,
        }}, {
        'name': 'Naive Bayes',
        'classifier': GaussianNB,
        }, {
        'classifier': KNeighborsClassifier,
        'name': 'Knn',
        'classifierParams': {
            'n_neighbors': 20,
            'weights': 'distance',
            'metric': 'euclidean',
        }}, {
        'classifier': RandomForestClassifier,
        'name': 'Random Forest',
        'classifierParams': {
            'criterion': 'gini',
             'max_depth': 30,

    }}
        ]

else:
    print('Unknown task: "{}" - Only options are [preprocessing] ,[unsupervised] or [classification]'.format(clArgs['TASK']))
    exit()

'''The following configurations usually are set via the jarvisConfig.yaml but we decided to skip that for the mooshak version
to avoid unnecessary complications.'''


print('> Importing puppet configuration from {}'.format(conf))
defaults = getConfiguration(conf)
defaults = selectFuncAccordingToParams(defaults, argListPuppet)

for c in changes:
    c['dataset'] = dataset
    c['numTrainingPoints'] = clArgs['numLines']

for c in changes:
    config = defaults.copy()
    config.update(c)
    print("==========        RUNNING TEST RUN - [{}]     ==========".format(config['name']))
    printDict(config, statement="> Using args:")

    # Create output directory for instance
    dir = getWorkDir({'outputDir': outputDir}, config['name'], completedText=successString)

    # Run instance
    puppet = Puppet(args=config, debug=False, outputDir=dir)
    puppet.pipeline()
    dumpConfiguration(config, dir, unfoldConfigWith=argListPuppet)
