import pandas as pd
from pyfpgrowth import find_frequent_patterns, generate_association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from src.libs.pattern_mining import *
from src.libs.evaluate import *
from src.libs.utils import *
from src.libs.balancing import *
from src.libs.treatment import *
from src.libs.plot import *
from libs.standardPlots import  *

from src.args import *


class Puppet:
    def __init__(self, args, debug, outputDir):
        """ I could pass everything from args to self right here to allow better interpretation
        of what parameters are needed fo use this class, but since we're using JARVIS that information
        can be seen in 'args.py' """
        self.args = args
        self.debug = debug
        self.outputDir = outputDir


    def pipeline(self):

        df, fixFunction = self._load()
        df = self._fullDatasetPreprocessing(df, fixFunction)
        y = df.loc[:, self.args['classname']]
        x = df.drop(columns=[self.args['classname']])

        if 'patternMining' in self.args.keys() and self.args['patternMining']:
            x, y = self.args['balancingStrategy'](x, y)
            df = x.join(y)
            self.evaluatePatternMining(self.patternMining(df,x, y))      # Since no optimization is needed no return is necessary

        elif 'clustering' in self.args.keys() and self.args['clustering']:
            self.cluster_method = self.linkFunctionToArgs('clusterFunction', 'clusterParams')
            self.evaluate_clustering(*self.do_clustering(df, x, y, {}))

        else:
            self.clf = defineClassifier(self.args['classifier'], self.args)
            # Run classifier
            #self.clf = self.linkFunctionToArgs('classifier','classifierParams')

            # todo - Dont know if i can/should balance a dataset with class non binary
            x, y = self.args['balancingStrategy'](x, y)

            if self.args['dataSplitMethod'] == 'split':
                print('Splitting dataset into train/test')
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=420)

                x_train, y_train, x_test, y_test, extraInfo = self._postSplitPreprocessing(x_train, y_train, x_test, y_test)

                print('Training with x: {}'.format(x_train.shape))
                r = self.evaluateClf(*self.trainClf(x_train, x_test, y_train, y_test, extraInfo))
                printResultsToJson(r, self.outputDir)

                cost = (r['sensitivity'] + r['specificity']) / 2

            elif self.args['dataSplitMethod'] == 'kfold':
                if 'kFold' in self.args.keys():
                    print('Splitting dataset into {} folds'.format(self.args['kFold']))
                    # kf = KFold(n_splits=self.args['kFold'])
                    kf = StratifiedKFold(n_splits=self.args['kFold'])
                    results = {}
                    it = 0
                    for train_index, test_index in kf.split(x, y):

                        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        print('After fold split x train state: {}'.format(x_train.shape))
                        print('After fold split x test state: {}'.format(x_test.shape))

                        x_train, y_train, x_test, y_test, extraInfo = self._postSplitPreprocessing(x_train, y_train, x_test,
                                                                                                   y_test)

                        print('Training with x: {}'.format(x_train.shape))
                        x_train, y_train, x_test, y_test, y_predict, extraInfo, y_predict_train = \
                            self.trainClf(x_train, x_test, y_train, y_test, extraInfo)

                        r = self.evaluateClf(x_train, y_train, x_test, y_test, y_predict, extraInfo, y_predict_train,
                                             iterator=it)

                        for key, val in r.items():
                            if key in results.keys():
                                results[key].append(val)
                            else:
                                results[key] = [val]
                        it += 1

                    finalResults = {}
                    for key, vals in results.items():
                        if (isinstance(vals[0], int) or isinstance(vals[0], float)):
                            valsWithoutNans = list(pd.Series(vals).fillna(method='ffill').fillna(method='bfill'))

                            finalResults[key] = sum(vals) / len(vals)
                            finalResults[key + '_kfoldVals'] = valsWithoutNans

                        printResultsToJson(finalResults, self.outputDir)
                    cost = (r['sensitivity'] + r['specificity']) / 2    # For optimization
                else:
                    print('kfold param required for kfold split method')
                    exit()

            return cost

    def do_clustering(self, df, x, y, extraInfo):
        print('--- Clustering ---')
        x = self.args['rescaler'](x)
        if self.cluster_method.__class__.__name__ == 'KPrototypes':
            self.cluster_method.fit(x, categorical = self.categorical_cols)
        else:
            self.cluster_method.fit(x)

        y_pred = self.cluster_method.labels_
        if self.cluster_method.__class__.__name__ == 'KMeans':
            extraInfo['inertia'] = self.cluster_method.inertia_ 

        return x, y, y_pred, extraInfo

    def evaluate_clustering(self, x, y, y_pred, extraInfo):
        print('--- Clustering Evaluation ---')
        results = cluster_metrics(x, y, y_pred)

        #if self.cluster_method.__class__.__name__ == 'DBSCAN':
        #    eps_plot(x, file = "eps.png")

        results.update(extraInfo)
        printResultsToJson(results, self.outputDir)


    def trainClf(self, x_train, x_test, y_train, y_test, extraInfo):

        # Training
        print('Fitting data...')
        self.clf.fit(x_train, y_train)

        y_predict = self.clf.predict(x_test)
        y_predict_train = self.clf.predict(x_train)

        return x_train, y_train, x_test, y_test, y_predict, extraInfo, y_predict_train

    def evaluateClf(self, x_train, y_train, x_test, y_test, y_predict, extraInfo, y_predict_train, iterator=None):
        print('--- Evaluation ---')
        # Calculate evaluation measures
        results = evaluate(self.clf, x_train, y_train, x_test, y_test, y_predict, y_predict_train)

        # Overfitting plot
        '''fig, ax = plt.subplots()
        data = self.args.copy()
        data.update(results)
        finalData = {}
        for key, vals in results.items():
            if (isinstance(vals[0], int) or isinstance(vals[0], float)):
                finalData[key] = vals
        y_types = [['accuracy', 'accuracyTrain']]
        x_type = ['index']
        multipleYsLinePlot(ax, pd.DataFrame.from_dict(finalData), y_types, x_type, colors=[], labels=[], joinYToLabel=None)
        '''

        if 'saveModel' in self.args.keys() and self.args['saveModel']:
            saveModel(self.clf, self.outputDir)

        print('-- Plotting --')
        # Plot Roc curve
        results2 = confMatrix(y_test, y_predict, self.outputDir, iterator=iterator)

        if self.args['dataset'] != 'src/data/covtype.data':
            # Plot confusion matrix
            rocCurve(y_test, y_predict, self.outputDir, iterator=iterator)

        '''if self.args['classifier'].__name__ == 'DecisionTreeClassifier':
            decision_tree_visualizer(self.clf, self.outputDir)
        '''
        # Output results
        results.update(extraInfo)
        results.update(results2)

        # Return what we'd like to minimize
        return results

    def patternMining(self, df, x, y):
        print('--- Pattern Mining ---')
        default_values = {
            'min_sup': 0.35,
            'min_conf': 0.9,
            'min_lift': 1.2,
            'iteratively_decreasing_support': True,
            'pattern_metric': "lift",
            'min_patterns': 30,
            'n': 3,
            'type': 'cut'
        }

        params = {**default_values, **self.args['miningParams']} if 'miningParams' in self.args and \
                                                                    self.args['miningParams'] != None else default_values
        
        # make flow here based on args (quick stuff)
        
        if self.args['featureFunction'] == chi2:
            x,_ = normalize(x)

        columns = SelectKBest(self.args['featureFunction'], k=self.args['nFeatures']).fit(x, y).get_support()
        new_x = x.loc[:,columns]
        #dummi_x = dummify(discretize(new_x, n = self.args['miningParams']['n'], type = self.args['typeMiningParams']))
        dummi_x = dummify(discretize(new_x, n = params['n'], type = params['type']))
        freqs = get_frequent_itemsets(dummi_x, minsup = params['min_sup'], \
            iteratively_decreasing_support = params['iteratively_decreasing_support'], minpatterns = params['min_patterns'])
        assoc_rules = get_association_rules(freqs, metric = params['pattern_metric'], min_conf = params['min_conf'], min_lift = params['min_lift'])
        #assoc_rules.to_csv(self.outputDir+"/assoc_rules.csv")
        #lab 6:
        #interesting_rules[(rules['antecedent_len']>=3 and rules['confidence'] >=0.9)][0:10]
        #for r in interesting_rules:
        #   print(f"confidence: {confidence} support: {support} lift: {lift})
        return assoc_rules
        
        
        # Fixme - a good question would be to ask how to calculate the amount of memory needed according to the dataset
        """       
        dfInTuples = list(df.itertuples(index=False, name=None))

        patterns = find_frequent_patterns(dfInTuples, self.args['miningParams']['supportThreshold'])
        print(patterns)
        rules = generate_association_rules(patterns, self.args['miningParams']['supportConfidence'])
        print(rules)
        """

    def _fullDatasetPreprocessing(self, df, fixFunction):
        # Data treatment - We only require to run this once and save this in a new csv file to save time

        df = (df
              .pipe(fixFunction)
              .pipe(fillNan, strategy=self.args['nanStrategy']))
        print('Initial data frame ', df.shape)

        if self.debug:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('USING ONLY 10% OF THE DATA')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
            df = df.iloc[:int(df.shape[0]*0.10), :]

        return df

    def _postSplitPreprocessing(self, x, y, xTest, yTest):
        print('Applying Pre-processing')

        # Scaling
        if 'rescaler' in self.args.keys() and type(self.args['rescaler']) != str:
            x, xTest = self.args['rescaler'](x, xTest)  # normalize/standardize ....


        dropedCols = None
        # Feature select
        if 'PCA' in self.args.keys() and self.args['PCA'] and 'percComponentsPCA' in self.args.keys():
            numComponents = int(self.args['percComponentsPCA']*x.shape[1])
            x = (x.
                 pipe(StandardScaler).   # fixme - it's called before as well, not sure when it should
                 pipe(applyPCA, numComponents))

        elif 'featureFunction' in self.args.keys() and 'featuresToKeepPercentage' in self.args.keys() \
                and self.args['featureFunction'] != '':
            print('Applying feature selection')

            if self.args['featureFunction'] is PCA:
                x, xTest = featSelectPCA(x, xTest, self.args['featuresToKeepPercentage'])
            else:
                x, dropedCols = getBestFeatures(x, y, self.args['featureFunction'], self.args['featuresToKeepPercentage'])
                xTest = xTest.drop(dropedCols, axis=1)
            print('x Train state: {}'.format(x.shape))
            print('Test state: {}'.format(xTest.shape))

        elif 'correlationThreshold' in self.args.keys() and self.args['correlationThreshold'] != 1:
            print('Applying corr threshold')
            x, dropedCols = dropHighCorrFeat(x, max_corr=self.args['correlationThreshold'])
            xTest = xTest.drop(dropedCols, axis=1)
            print('x Train state: {}'.format(x.shape))
            print('Test state: {}'.format(xTest.shape))

        # n = 'correlation threshold: {}'.format(self.args<['correlationThreshold'])
        # correlation_matrix(x, n, join(self.outputDir, 'Correlation Mattrix.png'), annotTreshold=20)

        print('Treatment done, final x Train state: {}'.format(x.shape))
        print('Treatment done, final x Test state: {}'.format(xTest.shape))

        extraInfo = {'dropedCols': dropedCols} if dropedCols is not None else {}

        return x, y, xTest, yTest, extraInfo

    def _load(self):
        '''
        # For when nan strategy chosen
        datasetName = self.args['dataset'].replace('.csv', '')
        df = pd.read_csv(self.args['dataset']) if self.args['nanStrategy'] == '' \
                else pd.read_csv(datasetName + '_{}{}'.format(self.args['nanStrategy'], '.csv'))
        '''

        if 'pd_speech_features' in self.args['dataset']:
            df = pd.read_csv(self.args['dataset'], header=1,  sep=',', decimal='.')
            self.categorical_cols = [0, 1] #id, gender
            fixFunction = fixDataSetSpeach
        elif 'covtype' in self.args['dataset']:
            df = pd.read_csv(self.args['dataset'], header=None, sep=',', decimal='.')
            self.categorical_cols = [i for i in range(10, 54, 1)]
            fixFunction = fixDataSetCov
        else:
            print('Unkown dataset')
            exit()
        return df, fixFunction



    def evaluatePatternMining(self, assoc_rules):
        #assoc_rules = assoc_rules[~(assoc_rules.antecedents.map(len) > 2)]

        results = {
            'support': assoc_rules['support'].tolist(),
            'lift': assoc_rules['lift'].tolist(),
            'confidence': assoc_rules['confidence'].tolist(),
            'avg_support': np.mean(assoc_rules['support']),
            'avg_lift': np.mean(assoc_rules['lift']),
            'avg_confidence': np.mean(assoc_rules['confidence'])
            }
        rules=assoc_rules.copy()
        rules=rules.sort_values('lift')
        rules.iloc[-10:].to_csv( self.outputDir + "/top_rules.csv")
        printResultsToJson(results, self.outputDir)
