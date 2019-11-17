from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier


def defineClassifier(clf, args):

    if clf is VotingClassifier:
        return createVotingEnsemble(clf, args['ensembleParams'],
                                  args['classifierParams']['clfParams'])

    elif clf is BaggingClassifier:
        '''
        if args['base_estimator'] == 'decisionTree':
            args['base_estimator'] = DecisionTreeClassifier()
        elif args['ensembleParams']['base_estimator'] == 'naiveBays':
            args['ensembleParams']['base_estimator'] = GaussianNB()
        elif args['ensembleParams']['base_estimator'] == 'KNN':
            args['ensembleParams']['base_estimator'] = KNeighborsClassifier()
        '''
        clf = BaggingClassifier(**args['ensembleParams'])
        return clf
    else:
        if 'classifierParams' in args.keys() and args['classifierParams'] is not None:
            clf = clf(**args['classifierParams'])
        else:
            clf = clf()
        return clf


def createVotingEnsemble(ensembleFunction, ensembleParams, clfParams):
    if ensembleFunction is VotingClassifier:
        clfs = []
        for i, c in enumerate(clfParams):
            if 'KNN' in c['name']:
                clfs.append((c['name'], KNeighborsClassifier(**{i:c[i] for i in c if i!='name'})))
            elif 'naiveBays' in c['name']:
                clfs.append((c['name'], GaussianNB()))
            elif 'decisionTree' in c['name']:
                clfs.append((c['name'], DecisionTreeClassifier(**{i:c[i] for i in c if i!='name'})))

        votingClf = VotingClassifier(estimators=clfs, **ensembleParams)
        return votingClf

