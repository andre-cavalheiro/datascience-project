from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier


def defineClassifier(clf, args):

    if clf is VotingClassifier:
        return createVotingEnsemble(clf, args['classifierParams']['ensembleParams'], args['classifierParams']['clfFunctions'],
                                  args['classifierParams']['clfParams'])

    elif clf is BaggingClassifier:
        if args['classifierParams']['estimator'] == 'KNN':
            estimator = KNeighborsClassifier
        elif args['classifierParams']['estimator'] == 'naiveBays':
            estimator = GaussianNB
        elif args['classifierParams']['estimator'] == 'decisionTree':
            estimator = DecisionTreeClassifier
        clf = BaggingClassifier(base_estimator=estimator(), **args['classifierParams']['ensembleParams'])
        return clf
    else:
        if 'classifierParams' in args.keys() and args['classifierParams'] is not None:
            clf = clf(**args['classifierParams'])
        else:
            clf = clf()
        return clf


def createVotingEnsemble(ensembleFunction, ensembleParams, clfFunctions, clfParams):
    if ensembleFunction is VotingClassifier:
        clfs = []
        for i, c in enumerate(clfFunctions):
            if 'KNN' in c:
                clfs.append((c, KNeighborsClassifier(**clfParams[i])))
            elif 'naiveBays' in c:
                clfs.append((c, GaussianNB()))
            elif 'decisionTree' in c:
                clfs.append((c, DecisionTreeClassifier ** clfParams[i]))

        votingClf = VotingClassifier(estimators=clfs, **ensembleParams)
        return votingClf

