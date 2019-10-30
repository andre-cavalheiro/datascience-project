import pandas as pd
from pyfpgrowth import find_frequent_patterns, generate_association_rules
from sklearn.model_selection import train_test_split, KFold

from src.libs.balancing import *
from src.libs.treatment import *
from src.libs.plot import *
from src.libs.evaluate import *
from src.args import *

class Puppet:
    def __init__(self, args, debug, outputDir):
        """ I could pass everything from args to self right here to allow better interpretation
        of what parameters are needed fo use this class, but since we're using JARVIS that information
        can be seen in 'args.py' """
        self.args = args
        self.debug = debug
        self.outputDir = outputDir
        if 'classifierParams' in self.args.keys() and self.args['classifierParams'] is not None:
            self.clf = self.args['classifier'](**self.args['classifierParams'])
        else:
            self.clf = self.args['classifier']()

    def pipeline(self):
        if self.args['patternMining']:
            df, _, _, _ = self._treatment()
            self.patternMining(df)      # Since no optimization is needed no return is necessary

        elif self.args['clustering']:
            pass
        else:
            # Run classifier
            df, x, y, extraInfo = self._treatment()

            if self.args['dataSplitMethod'] == 'split':
                print('Splitting dataset into train/test')
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=420)

                print('Training with x: {}'.format(x.shape))
                print('Current x state: ', x.shape)
                
                r = self.evaluateClf(*self.trainClf(df, x_train, x_test, y_train, y_test, extraInfo))
                printResultsToJson(r, self.outputDir)
                
                cost = (r['sensitivity'] + r['specificity']) / 2

            elif self.args['dataSplitMethod'] == 'kfold':
                if 'kFold' in self.args.keys():
                    print('Splitting dataset into k folds')
                    kf = KFold(n_splits=self.args['kFold'])
                    results = {}
                    for train_index, test_index in kf.split(x):
                        # print("TRAIN:", train_index, "TEST:", test_index)
                        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        x_train, y_train, x_test, y_test, y_predict, extraInfo, y_predict_train = \
                            self.trainClf(df, x_train, x_test, y_train, y_test, extraInfo)

                        r = self.evaluateClf(x_train, y_train, x_test, y_test, y_predict, extraInfo, y_predict_train)

                        for key, val in r.items():
                            if key in results.keys():
                                results[key].append(val)
                            else:
                                results[key] = [val]

                    averagedResults = {}
                    for key, vals in results.items():
                        if(isinstance(vals[0], int) or isinstance(vals[0], float)):
                            averagedResults[key] = sum(vals)/len(vals)

                    printResultsToJson(averagedResults, self.outputDir)
                    cost = (r['sensitivity'] + r['specificity']) / 2
                else:
                    print('kfold param required for kfold split method')
                    exit()
        return cost

    def _treatment(self):

        # Data load

        '''
        # For when nan strategy chosen
        datasetName = self.args['dataset'].replace('.csv', '')
        df = pd.read_csv(self.args['dataset']) if self.args['nanStrategy'] == '' \
                else pd.read_csv(datasetName + '_{}{}'.format(self.args['nanStrategy'], '.csv'))
        '''
        if 'pd_speech_features' in self.args['dataset']:
            df = pd.read_csv(self.args['dataset'], header=1,  sep=',', decimal='.')
            fixFunction = fixDataSetSpeach
        elif 'covtype' in self.args['dataset']:
            df = pd.read_csv(self.args['dataset'], header=None, sep=',', decimal='.')
            fixFunction = fixDataSetCov

        # Data treatment - We only require to run this once and save this in a new csv file to save time

        df = (df
              .pipe(fixFunction)
              .pipe(fillNan, strategy=self.args['nanStrategy']))
        print('Initial data frame ', df.shape)

        if self.debug:
            print('USING ONLY 10% OF THE DATA')
            df = df.iloc[:int(df.shape[0]*0.10), :]
        
        y = df.loc[:, self.args['classname']]
        x = df.drop(columns=[self.args['classname']])

        dropedCols = None
        if 'PCA' in self.args.keys() and self.args['PCA']:
            numComponents = int(self.args['percComponentsPCA']*x.shape[1])
            x = applyPCA(x, numComponents)

            x = (x.
                 pipe(self.args['rescaler']).   # normalize/standardize ....
                 pipe(applyPCA, numComponents))

        else:
            if self.args['correlationThreshold'] != 1:
                x, dropedCols = dropHighCorrFeat(x, max_corr=self.args['correlationThreshold'])
            x = self.args['rescaler'](x)    # normalize/standardize ....


        print('Treatment done, final x state: {}'.format(x.shape))

        # n = 'correlation threshold: {}'.format(self.args<['correlationThreshold'])
        # correlation_matrix(x, n, join(self.outputDir, 'Correlation Mattrix.png'), annotTreshold=20)

        # todo - dont know if i can/should balance a dataset with class non binary
        if 'pd_speech_features' in self.args['dataset']:
            x, y = self.args['balancingStrategy'](x, y)

        df = x.join(y)

        extraInfo = {'dropedCols': dropedCols} if dropedCols is not None else {}
        return df, x, y, extraInfo

    def trainClf(self, df, x_train, x_test, y_train, y_test, extraInfo):

        # Training
        print('Fitting data...')
        self.clf.fit(x_train, y_train)

        y_predict = self.clf.predict(x_test)
        y_predict_train = self.clf.predict(x_train)

        return x_train, y_train, x_test, y_test, y_predict, extraInfo, y_predict_train

    def evaluateClf(self, x_train, y_train, x_test, y_test, y_predict, extraInfo, y_predict_train):
        print('--- Evaluation ---')
        # Calculate evaluation measures
        results = evaluate(self.clf, x_train, y_train, x_test, y_test, y_predict, y_predict_train)


        if 'saveModel' in self.args.keys() and self.args['saveModel']:
            saveModel(self.clf, self.outputDir)

        print('-- Plotting --')
        # Plot Roc curve
        results2 = confMatrix(y_test, y_predict, self.outputDir)

        if self.args['dataset'] != 'src/data/covtype.data':
            # Plot confusion matrix
            rocCurve(y_test, y_predict, self.outputDir)

        # Output results
        results.update(extraInfo)
        results.update(results2)


        # Return what we'd like to minimize
        return results

    def patternMining(self, df):
        # Mine por frequent patterns and find association rules
        # Fixme - a good question would be to ask how to calculate the amount of memory needed according to the dataset
        dfInTuples = list(df.itertuples(index=False, name=None))

        patterns = find_frequent_patterns(dfInTuples, self.args['miningParams']['supportThreshold'])
        print(patterns)
        rules = generate_association_rules(patterns, self.args['miningParams']['supportConfidence'])
        print(rules)
