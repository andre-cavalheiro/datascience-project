import argparse
from os.path import join
from os import walk
import random, string
import json
import pandas as pd
import yaml
import copy
from libs.dir import *
from libs.standardPlots import *
from src.puppet import Puppet
# Import yaml
def getConfiguration(configFile):
    with open(configFile, 'r') as stream:
        params = yaml.safe_load(stream)
    return params

# Dump yaml
def dumpConfiguration(configDict, direcotry, unfoldConfigWith):
    if unfoldConfigWith:
        configDict = selectParamsAccordingToFunctions(configDict, unfoldConfigWith) # Reverse functions to its arg names
    t = join(direcotry, 'config.yaml')
    with open(t, 'w') as f:
        yaml.dump(configDict, f, default_flow_style=False)


# Hack for non-default
class NonDefaultVerifier(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest+'_nonDefault', True)


#  Dynamic boolean type for argument parser
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def randomName(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))


def printDict(dict, statement=None):
    if statement:
        print(statement)
    for attr, value in sorted(dict.items()):
        print("\t{}={}".format(attr.upper(), value))


def selectFuncAccordingToParams(config, argList):
    config = copy.deepcopy(config)
    for a in argList:
        if 'possibilities' in a.keys() and len(a['possibilities']) is not 0:
            for p in a['possibilities']:
                if a['name'] in config.keys() and  p[0] == config[a['name']]:
                    config[a['name']] = p[1]
                    break
    return config


def selectParamsAccordingToFunctions(config, argList):
    for a in argList:
        if 'possibilities' in a.keys() and len(a['possibilities']) is not 0:
            for p in a['possibilities']:
                if a['name'] in config.keys() and p[1] == config[a['name']]:
                    config[a['name']] = p[0]
                    break

        '''if 'children' in a.keys() and len(a['children']) is not 0:
            if a['name'] == 'ensembleParams':
            for c in a['children']:
                if 'possibilities' in c.keys() and len(c['possibilities']) is not 0:
                    for pp in c['possibilities']:
                        if c['name'] in config[a['name']].keys() and pp[1] == config[a['name']][c['name']]:
                            config[a['name']][c['name']] = pp[0]
                            break
        '''
    return config

def getTrialValuesFromConfig(trial, pconfig, argListPuppet):
    for arg in argListPuppet:
        trialVal = getTrialValues(trial, arg)
        pconfig[arg['name']] = trialVal if trialVal is not None else pconfig[arg['name']]
        if 'children' in arg.keys() and pconfig[arg['name']] is not None:
            for childArg in arg['children']:
                trialVal = getTrialValues(trial, childArg)
                pconfig[arg['name']][childArg['name']] = trialVal if trialVal is not None \
                    else pconfig[arg['name']][childArg['name']]

    return pconfig

def getTrialValues(trial, arg):
    if 'optimize' in arg.keys() and arg['optimize']:
        if 'optimizeInt' in arg.keys():
            return trial.suggest_int(arg['name'], arg['optimizeInt'][0], \
                                                     arg['optimizeInt'][1])

        elif 'optimizeUniform' in arg.keys():
            return trial.suggest_uniform(arg['name'], arg['optimizeUniform'][0], \
                                                         arg['optimizeUniform'][1])

        elif 'optimizeLogUniform' in arg.keys():
            return trial.suggest_loguniform(arg['name'], arg['optimizeLogUniform'][0], \
                                                            arg['optimizeLogUniform'][1])

        elif 'optimizeDiscreteUniform' in arg.keys():
            return trial.suggest_loguniform(arg['name'], *arg['optimizeDiscreteUniform'])

        elif 'optimizeCategorical' in arg.keys():
            return trial.suggest_categorical(arg['name'], arg['optimizeCategorical'])
        else:
            return None
    else:
        return None


# Evaluate if situation demands for a plot, and if so apply the ones that are in order.
def makePrettyPlots(config, dir, possibilities, unify=False, logFile='logs.json',
                    configFile='config.yaml', unificationType=''):


    # Some plots require the unification of outputs.
    unifyOutputs(unify, unificationType, logFile, configFile, dir)

    assert(len(config['ys']) == len(config['type']))

    types, funcs, args = [], [], []
    for p in possibilities:
        types.append(p[0])
        funcs.append(p[1])
        args.append(p[2])
    # For every enabled mode make the plot
    for t, type in enumerate(config['type']):
        typeId = types.index(type)
        xSpecificType=config['x'][t]
        ysSpecificType=config['ys'][t]
        # Build common params between iterations and remove them from the other ones
        sharedParams = {a[0]: config[a[0]] for a in args[typeId] if a[1] == 'shared' if a[0] in config.keys()}
        toDeleteInd = [i for i, a in enumerate(args[typeId]) if a[0] in sharedParams.keys()]
        for index in sorted(toDeleteInd, reverse=True):
            del args[typeId][index]

        # Plot
        for j in range(len(xSpecificType)):
            # Build iteration specific params

            params = {n[0]: config[n[0]][j] for n in args[typeId] if n[0] in config.keys()}
            funcs[typeId](x=xSpecificType[j], ys=ysSpecificType[j], dpi=config['dpi'], level=config['level'],
                          dir=dir, **params, **sharedParams)


def unifyOutputs(unify, unificationType, logFile, configFile, dir):
    if unify:
        if unificationType == 'yaml,json-Csv':
            unifyJsonYamlOutputsIntoCSV(dir, logFile=logFile, configFile=configFile)
        else:
            print('Unkown unification type {}'.format(unificationType))
            exit()

# Unify several config.yamls and and logs.json into a single output.csv for the overall sequential running
def unifyJsonYamlOutputsIntoCSV(dir, logFile='logs.json', configFile='config.yaml'):
    logType = logFile.split('.')[-1]
    datasetCols = {}

    for subdir, dirs, files in walk(dir):
        for subdir in dirs:
            # todo - Check name to avoid sequentials and optimizations
            subpathToConf = join(dir, subdir, configFile)
            subpathToOutput = join(dir, subdir, logFile)

            conf = getConfiguration(subpathToConf)
            conf = removeNestedDict(conf)

            with open(subpathToOutput) as f:
                if logType == 'json':
                    outputs = json.load(f)
                elif logType == 'csv':
                    print('not implemented format csv')
                    pass
                else:
                    print('Unknown format {}'.format(logType))

            conf.update(outputs)

            newAddOns = []

            for key, val in conf.items():
                if key != 'dropedCols':  # fixme
                    if key in datasetCols.keys():
                        datasetCols[key].append(val)
                    else:
                        newAddOns.append((key, val))

            for t in newAddOns:
                datasetCols[t[0]] = [t[1]]

    unified_results = pd.DataFrame(data=datasetCols)
    unified_results.to_csv("{}/output.csv".format(dir), sep=',', encoding='utf-8')

    return unified_results


def removeNestedDict(d):
    newAddOns = []
    for key, val in d.items():
        if isinstance(val, dict):
            for key_, val_ in val.items():
                newAddOns.append((key, key_, val_))
    for t in newAddOns:
        if t[0] in d.keys():
            d.pop(t[0])
        d[t[1]] = t[2]
    return d


def makePlotConf(plotConfig, paramType, pconfig):
    a = plotConfig.copy()
    b = plotConfig[paramType]
    del a['plotSingleParams']
    del a['plotSeqParams']
    del a['plotOnlyParams']
    a.update(b)
    a['x'] = [[pconfig['plotX']for k in y] for y in a['ys']]
    return a

def recursivelyBuildConfigs(pipeline, configs, argListPuppet, iterator=0):
    pconfigs= []

    if len(pipeline)-1 == iterator:
        p = pipeline[0]
        for var in p['values']:
            if 'subName' in p.keys():
                if p['name'] in configs.keys():
                    configs[p['name']][p['subName']] = var
                else:
                    configs[p['name']] = {}
                    configs[p['name']][p['subName']] = var
            else:
                configs[p['name']] = var

            pconfig = selectFuncAccordingToParams(configs, argListPuppet)
            pconfigs.append(pconfig)

        return pconfigs

    for i in range(iterator, len(pipeline)):
        p = pipeline[i]
        for var in p['values']:
            c = configs.copy()
            if 'subName' in p.keys():
                c[p['name']][p['subName']] = var
            else:
                c[p['name']] = var

            pconfig = recursivelyBuildConfigs(pipeline[i + 1:], c, argListPuppet, iterator)
            pconfigs += pconfig
        break

    return pconfigs


def recursivelyRunPuppets(pipeline, iterator, pconfigs, prevDir, jconfig,
                    plotConfig, argListPuppet, argListPlots, seqConf, names, numRan=0):


    if len(pipeline)-1 == iterator:
        for i, var in enumerate(pipeline[iterator]):
            pconfig = pconfigs[numRan]

            print("=== NEW INSTANCE ==  ")
            # printDict(pconfig, statement="> Using args:")
            # Create output directory for instance inside sequential-test directory
            n = names[iterator] + ' - {}'.format(var) if type(var) != type([]) else  names[iterator] + ' - {}'.format(var[0]['name'])
            dir = makeDir(copy.copy(prevDir), n, completedText=jconfig['successString'])

            puppet = Puppet(copy.copy(pconfig), debug=jconfig['debug'], outputDir=dir)
            puppet.pipeline()
            dumpConfiguration(copy.copy(pconfig), dir, unfoldConfigWith=argListPuppet)
            numRan+=1
            if 'single' in jconfig['plot']:
                # Get plot possibilities for selected mode
                currentPlotConf = makePlotConf(plotConfig, 'plotSingleParams', seqConf)
                g = (e for e in argListPlots if e.get('name') == 'singlePlotTypes')
                plotTypes = next(g)
                # fixme - should not be harde coded

                currentPlotConf['ys'] = currentPlotConf['ysSingle']
                currentPlotConf['x'] = [[currentPlotConf['xSingle'] for p in currentPlotConf['ys']]]
                makePrettyPlots(currentPlotConf, dir, plotTypes['possibilities'], unify=True, logFile='logs.json',
                                configFile='config.yaml', unificationType=plotConfig['seqLogConversion'])

            changeDirName(dir, extraText=jconfig['successString'])

        if len(pipeline)==1 and 'seq' in jconfig['plot']:
            # Get plot possibilities for selected mode
            currentPlotConf = makePlotConf(plotConfig, 'plotSeqParams', seqConf)
            g = (e for e in argListPlots if e.get('name') == 'seqPlotTypes')
            plotTypes = copy.deepcopy(next(g))
            un = False
            if seqConf['unifyByRecursionLevels'][iterator] == 1:
                un = True
            makePrettyPlots(currentPlotConf, prevDir, plotTypes['possibilities'], unify=un,
                            logFile='logs.json', configFile='config.yaml',
                                unificationType=plotConfig['seqLogConversion'])
            changeDirName(prevDir, extraText=jconfig['successString'])

        return numRan
    else:
        for i, var in enumerate(pipeline[iterator]):

            dir = getWorkDir({'outputDir': prevDir}, '{} - {}'.format(names[iterator], var))
            numRan = recursivelyRunPuppets(pipeline, iterator+1, pconfigs, dir, jconfig,
                                  plotConfig, argListPuppet, argListPlots, seqConf, names, numRan)

            if 'seq' in jconfig['plot']:
                # Get plot possibilities for selected mode
                currentPlotConf = makePlotConf(plotConfig, 'plotSeqParams', seqConf)
                g = (e for e in argListPlots if e.get('name') == 'seqPlotTypes')
                plotTypes = copy.deepcopy(next(g))
                un = False

                if seqConf['unifyByRecursionLevels'][iterator] == 1:
                    un = True

                if 'changedToLowerDimPlot' in seqConf.keys():
                    for k, v in seqConf['changedToLowerDimPlot'].items():
                        currentPlotConf[k] = v

                makePrettyPlots(currentPlotConf, dir, plotTypes['possibilities'], unify=un,
                                logFile='logs.json', configFile='config.yaml',
                                unificationType=plotConfig['seqLogConversion'])

            changeDirName(dir, extraText=jconfig['successString'])

        if 'seq' in jconfig['plot']:
            # Get plot possibilities for selected mode
            currentPlotConf = makePlotConf(plotConfig, 'plotSeqParams', seqConf)
            g = (e for e in argListPlots if e.get('name') == 'seqPlotTypes')
            plotTypes = copy.deepcopy(next(g))
            un = False  # fixme - should not be hardecoded
            if 'changedToHigherDimPlot' in seqConf.keys():
                for k, v in seqConf['changedToHigherDimPlot'].items():
                    currentPlotConf[k] = v
            makePrettyPlots(currentPlotConf, prevDir, plotTypes['possibilities'], unify=un,
                            logFile='logs.json', configFile='config.yaml',
                            unificationType=plotConfig['seqLogConversion'])

        changeDirName(prevDir, extraText=jconfig['successString'])
        return numRan
