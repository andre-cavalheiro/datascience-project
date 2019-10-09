import pandas as pd
from pyfpgrowth import find_frequent_patterns, generate_association_rules
from sklearn.model_selection import train_test_split

from src.libs.balancing import *
from src.libs.treatment import *
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

        if self.args['clustering']:
            pass
        else:
            # Run classifier
            return self.evaluateClf(*self.trainClf(*self._treatment()))

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

        if 'PCA' in self.args.keys() and self.args['PCA']:
            numComponents = int(self.args['percComponentsPCA']*x.shape[1])
            x = applyPCA(x, numComponents)

            x = (x.
                 pipe(self.args['rescaler']).   # normalize/standardize ....
                 pipe(applyPCA, numComponents))

        else:
            x, dropedCols = dropHighCorrFeat(x, max_corr=self.args['covarianceThreshold'])
            x = self.args['rescaler'](x)    # normalize/standardize ....

        print('Treatment done, final x state: {}'.format(x.shape))

        df = x.join(y)

        return df, x, y, {dropedCols: dropedCols}

    def trainClf(self, df, x, y, extraInfo):

        print('Splitting dataset into train/test')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=420)

        print('Training with x: {}'.format(x.shape))
        print('Current x state: ', x.shape)

        # todo - dont know if i can/should balance a dataset with class non binary
        if 'pd_speech_features' in self.args['dataset']:
            x_train, y_train = self.args['balancingStrategy'](x_train, y_train)

        # Training
        print('Fitting data...')
        self.clf.fit(x_train, y_train)
        y_predict = self.clf.predict(x_test)

        return x_train, y_train, x_test, y_test, y_predict

    def evaluateClf(self, x_train, y_train, x_test, y_test, y_predict):
        print('--- Evaluation ---')
        # Calculate evaluation measures
        results = evaluate(self.clf, x_train, y_train, x_test, y_test, y_predict)

        if 'saveModel' in self.args.keys() and self.args['saveModel']:
            saveModel(self.clf, self.outputDir)

        print('-- Plotting --')
        # Plot Roc curve
        results2 = confMatrix(y_test, y_predict, self.outputDir)
        # Plot confusion matrix
        rocCurve(y_test, y_predict, self.outputDir)

        # Output results
        results.update(extraInfo)
        results.update(results2)
        printResultsToJson(results, self.outputDir)

        cost = (results['sensitivity']+results['specificity'])/2

        # Return what we'd like to minimize
        return -cost

    def patternMining(self, df):
        # Mine por frequent patterns and find association rules
        # Fixme - a good question would be to ask how to calculate the amount of memory needed according to the dataset
        dfInTuples = list(df.itertuples(index=False, name=None))

        patterns = find_frequent_patterns(dfInTuples, self.args['miningParams']['supportThreshold'])
        print(patterns)
        rules = generate_association_rules(patterns, self.args['miningParams']['supportConfidence'])
        print(rules)
