import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, roc_curve, auc, \
    confusion_matrix, log_loss, cohen_kappa_score
from sklearn.model_selection import cross_val_score, learning_curve
from os.path import join, exists, isfile
import joblib

def evaluate(classifier, x_train, y_train, x_test, y_test, y_predict, y_predict_train=None, iterator=None):

    # Calculate
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='micro')
    recall = recall_score(y_test, y_predict, average='micro')
    f1 = f1_score(y_test, y_predict, average='micro')
    kappa = cohen_kappa_score(y_test, y_predict)

    attrs = ['accuracy', 'precision', 'recall', 'f1', 'kappa']
    values = [accuracy, precision, recall, f1, kappa]
    if y_predict_train is not None:
        accuracyTrain = accuracy_score(y_train, y_predict_train)
        attrs.append('accuracyTrain')
        values.append(accuracyTrain)

    # Log results
    logs = {attrs[it]: val for it, val in enumerate(values)}

    # Print
    for attr, value in sorted(logs.items()):
        print("\t{}: {}".format(attr.upper(), value))

    return logs


def printResultsToJson(logs, dir):
    with open(join(dir, 'logs.json'), 'w') as outfile:
        json.dump(logs, outfile)


def saveModel(model, dir):
    joblib.dump(model, join(dir, 'model.joblib'))


def loadModel(model, dir):
    model = joblib.load(model, join(dir, 'model.joblib'))
    return model


def confMatrix(y_test, y_predict, dir, iterator=None):

    confusionMatrix = confusion_matrix(y_test, y_predict)

    # Calculate some parameters depending on TP, FP, FN, TN
    sensitivity = confusionMatrix[0, 0]/(confusionMatrix[0, 0]+confusionMatrix[1, 0])
    specificity = confusionMatrix[1, 1]/(confusionMatrix[1, 1]+confusionMatrix[0, 1])
    recall = confusionMatrix[1, 1]/(confusionMatrix[1, 1]+confusionMatrix[1, 0])
    print('Sensitivity: %f' % sensitivity)
    print('Specificity: %f' % specificity)
    print('recall ', recall)

    # fixme - find cleaner way to do this ?
    # Log results
    attrs = ['sensitivity', 'specificity']
    values = [sensitivity, specificity]
    logs = {attrs[it]: val for it, val in enumerate(values)}

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(confusionMatrix, cmap=plt.cm.Blues, alpha=0.4)
    for i in range(confusionMatrix.shape[0]):
        for j in range(confusionMatrix.shape[1]):
            ax.text(x=j, y=i,
                    s=confusionMatrix[i, j],
                    va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    # plt.show()
    name = join(dir, 'conf-mat.png') if iterator==None else join(dir, 'conf-mat-{}.png'.format(iterator) )
    plt.savefig(name)
    print('Plotted confusion matrix')
    plt.close(fig=fig)
    return logs




def rocCurve(y_test, y_predict, dir, iterator=None):
    # Roc score (area under the curve)
    roc_auc_score_ = roc_auc_score(y_test, y_predict, )
    print('Roc auc score = ', roc_auc_score_)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
    roc_auc = auc(fpr, tpr)  # Area under the roc curve

    # Plot of a ROC curve for a specific class
    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve')
    plt.legend(loc="lower right")
    # plt.show()
    name = join(dir, 'roc.png') if iterator==None else join(dir, 'roc-{}.png'.format(iterator))
    plt.savefig(name)
    print('Plotted roc curve')
    plt.close(fig=fig)



def plot_learning_curve(estimator, X, y, dir, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title('learning curve')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    # plt.show()

    plt.savefig(join(dir, 'learning.png'))
    print('Plotted learning curve')


def plotParamVariationPretty(x, y1, y2, y3, label1='', label2='', label3='', xlabel='', ylabel='', title='', path='.'):
    fig, ax1 = plt.subplots()
    ax1.grid(which='major', alpha=0.3)

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.plot(x, y1, label=label1)
    ax1.plot(x, y2, label=label2)

    if y3 != None:
        ax2 = ax1.twinx()
        ax2.plot(x, y3, label=label3, color='green')
        ax2.tick_params(axis='y', labelcolor='green')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    print('----------------------------------------------------------------------------------------------------------')
    plt.legend()
    plt.savefig(path)
    plt.close(fig=fig)



def plotEvolutionaryPlots(x, y1, y2, y3=None, clf_family=None):
    if y3 == None:
        label3=None
    else:
        label3='cost'
    if clf_family == "knn_un":
        plotParamVariationPretty(x, y1, y2, y3, label1='recall', label2='precision', label3=label3,
                                 xlabel='Number of estimators', \
                                 ylabel='K value', title='KNN uniform weights - k tuning', path='./media/' + clf_family)

    if clf_family == "knn_dist":
        plotParamVariationPretty(x, y1, y2, y3, label1='recall', label2='precision', label3=label3, xlabel='K value', \
                                 ylabel='Performance', title='KNN distributed weights - K tunning',
                                 path='./media/' + clf_family)

    if clf_family == "rf_entr":
        plotParamVariationPretty(x, y1, y2, y3, label1='recall', label2='precision', label3=label3, xlabel='Depth', \
                                 ylabel='Performance', title='Random Forest entropy - depth tuning',
                                 path='./media/' + clf_family)

    if clf_family == "rf_gini":
        plotParamVariationPretty(x, y1, y2, y3, label1='recall', label2='precision', label3=label3, xlabel='Depth', \
                                 ylabel='Performance', title='Random Forest gini - depth tuning',
                                 path='./media/' + clf_family)

    if clf_family == "rf_gini_n":
        plotParamVariationPretty(x, y1, y2, y3, label1='recall', label2='precision', label3=label3,
                                 xlabel='nº estimators', \
                                 ylabel='Performance', title='Random Forest Gini - nº estimators tunning',
                                 path='./media/' + clf_family)

