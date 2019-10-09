import yaml
import argparse
from os.path import join, exists, isfile, isdir, basename, normpath
from os import makedirs, listdir, rename, getcwd, walk
import random, string
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

# Hack for non-default
class NonDefaultVerifier(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest+'_nonDefault', True)


# Import yaml
def getConfiguration(configFile):
    with open(configFile, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


# Dump yaml
def dumpConfiguration(configDict, direcotry, unfoldConfigWith=None):
    if unfoldConfigWith:
        configDict = selectParamsAccordingToFunctions(configDict, unfoldConfigWith) # Reverse functions to its arg names
    t = join(direcotry, 'config.yaml')
    with open(t, 'w') as f:
        yaml.dump(configDict, f, default_flow_style=False)


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


def getWorkDir(jconfig, folderName, extraName='', completedText='', createNew=True):
    if 'outputDir' in jconfig.keys() and jconfig['outputDir'] is not None:
        outputDir = jconfig['outputDir']
        return makeDir(outputDir, folderName + extraName, createNew=createNew, completedText=completedText)
    else:
        return getcwd()      # Current working directory


# fixme - this one is a bit ugly and can be improved
def makeDir(outputDir, name, createNew=True, completedText=''):
    dir = join(outputDir, name)
    dirCompleted = join(outputDir, name+completedText)
    if exists(dir) or exists(dirCompleted):
        j = 1
        while exists(dir) or exists(dirCompleted):
            j += 1
            dir = join(outputDir, '{}-{}'.format(name, j))
            dirCompleted = join(outputDir, '{}-{}{}'.format(name, j, completedText))

        if createNew:
            print('> Wanted directory already exists, created - "{}"'.format(dir))
            makedirs(dir)
        else:
            dir = join(outputDir, '{}-{}'.format(name, j-1)) if j != 2 else join(outputDir, name)
            print('> Using already existent dir - "{}"'.format(dir))
    else:
        print('> Created Directory - "{}"'.format(dir))
        makedirs(dir)
    return dir


def changeDirName(origPath, nemName='', extraText=''):
    if not isdir(origPath):
        print('Error: Given Path is not a directory, name unchanged')
        return
    pathToFolder = Path(origPath).parent
    oldFoderName = basename(normpath(origPath))
    newFolderName = oldFoderName + extraText

    rename(origPath, join(pathToFolder, newFolderName))


def getFilesInDir(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    return files


def printDict(dict, statement=None):
    if statement:
        print(statement)
    for attr, value in sorted(dict.items()):
        print("\t{}={}".format(attr.upper(), value))


def selectFuncAccordingToParams(config, argList):
    for a in argList:
        if 'possibilities' in a.keys() and len(a['possibilities']) is not 0:
            for p in a['possibilities']:
                if p[0] == config[a['name']]:
                    config[a['name']] = p[1]
                    break
    return config


def selectParamsAccordingToFunctions(config, argList):
    for a in argList:
        if 'possibilities' in a.keys() and len(a['possibilities']) is not 0:
            for p in a['possibilities']:
                if p[1] == config[a['name']]:
                    config[a['name']] = p[0]
                    break
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


def unifyOutputs(dir):
    datasetCols = {}
    print(dir)
    for subdir, dirs, files in walk(dir):
        for subdir in dirs:
            # todo - Check name to avoid sequentials and optimizations
            subpathToOutput = join(join(dir,subdir), 'logs.json')
            subpathToConf = join(join(dir,subdir), 'config.yaml')

            with open(subpathToOutput) as f:
                outputs = json.load(f)

            conf = getConfiguration(subpathToConf)
            conf = removeNestedDict(conf)

            conf.update(outputs)

            newAddOns = []

            for key, val in conf.items():
                if key in datasetCols.keys():
                    datasetCols[key].append(val)
                else:
                    newAddOns.append((key, val))

            for t in newAddOns:
                datasetCols[t[0]] = [t[1]]

    unified_results = pd.DataFrame(data=datasetCols)
    unified_results.to_csv("{}/output.csv".format(dir), sep='\t', encoding='utf-8')

    return unified_results 


def plot_output(file, y_types, x_type = None):
    data = pd.read_csv(file,  sep='\t', encoding='utf-8')
    fig, ax = plt.subplots()
    
    if x_type == None or x_type == "index":
        x = [str(v) for v in data.index.values]
    else:
        data = data.sort_values(by=[x_type])
        x = data[x_type]

    print(data)
    for t in y_types:
        ax.plot(x, data[t], label=t)

    ax.legend()
    plt.show()

plot_output("src/output/optimization/output.csv",["sensitivity","specificity"],  "covarianceThreshold")