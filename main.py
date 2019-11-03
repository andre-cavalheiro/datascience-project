from sys import exit
from libs.utils import *
from libs.dir import *
from libs.standardPlots import *
from src.puppet import Puppet
from src.args import argListPuppet
from argsConf.jarvisArgs import argListJarvis
from argsConf.plotArgs import argListPlots
from os.path import join
import optuna
import copy

# Verify command line arguments
parser = argparse.ArgumentParser(description='[==< J A R V I S >==]')
for arg in argListJarvis:
    parser.add_argument('-{}'.format(arg['name']), type=arg['type'], default=arg['default'], required=arg['required'],
                        help=arg['help'], action=NonDefaultVerifier)
for arg in argListPuppet:
    parser.add_argument('-{}'.format(arg['name']), type=arg['type'], default=arg['default'], required=False,
                        help=arg['help'], action=NonDefaultVerifier)
clArgs = parser.parse_args()
argsPassedJarvis = [a['name'] for a in argListJarvis if hasattr(clArgs, '{}_nonDefault'.format(a['name']))]     # Find which params were actually passed
argsPassedPuppet = [a['name'] for a in argListPuppet if hasattr(clArgs, '{}_nonDefault'.format(a['name']))]     # Find which params were actually passed
clArgs = vars(clArgs)      # Convert to dictionary

# Import jarvis configurations if it exists
if clArgs['jc']:
    print('> Importing configuration from {}'.format(clArgs['jc']))
    jconfig = getConfiguration(clArgs['jc'])
    # Upgrade values if command line ones were passed
    for key in argsPassedJarvis:
        jconfig[key] = clArgs[key]
# If no config file then use only command line args
else:
    # Check if all required arguments have been passed
    jconfig = {}
    for arg in argListJarvis:
        if arg['required'] and arg['name'] not in argsPassedJarvis:
            print('> Missing required JARVIS argument "{}" - exiting'.format(arg['name']))
            exit()
        else:
            jconfig[arg['name']] = clArgs[arg['name']]

# Attribute random name to test run if one wasn't provided
if 'name' not in jconfig.keys() or ('name' in jconfig.keys() and jconfig['name'] is None):
    jconfig['name'] = randomName(7)

printDict(jconfig, statement="> JARVIS using args:")

# Import plot args:
if 'confPlot' in jconfig.keys():
    plotConfig = getConfiguration(jconfig['confPlot'])

if 'optimizer' in jconfig.keys() and 'optimize' in jconfig.keys() and jconfig['optimize']:

    # Create working directory here since it cannot be inside optimization function
    optimizationDir = getWorkDir(jconfig, 'optimization - {}'.format(jconfig['name']), completedText=jconfig['successString'])

    def optimizationObjective(trial):
        # fixme - Lots of unnecessary accesses to the file
        print("=== NEW TRIAL ==  ")

        pconfig = getConfiguration(jconfig['conf'])

        for arg in argListPuppet:
            if arg['name'] not in pconfig.keys():
                pconfig[arg['name']] = None         # todo This may be needed somewhere else
        pconfig = getTrialValuesFromConfig(trial, pconfig, argListPuppet)
        printDict(pconfig, "> Trial for: ")
        pconfig = selectFuncAccordingToParams(pconfig, argListPuppet)

        # Get working directory name that was created in the beginning of the optimization procedure.
        optimizationDir = getWorkDir(jconfig, 'optimization - {}'.format(jconfig['name']), createNew=False, completedText=jconfig['successString'])
        dir = makeDir(optimizationDir, 'trial', completedText=jconfig['successString'])

        puppet = Puppet(pconfig, debug=jconfig['debug'], outputDir=dir)
        reward = puppet.pipeline()
        dumpConfiguration(pconfig, dir, unfoldConfigWith=argListPuppet)
        changeDirName(dir, extraText=jconfig['successString'])

        return reward

    print("==========        RUNNING OPTIMIZATION FOR - [{}]     ==========".format(jconfig['name']))

    study = optuna.create_study(study_name=jconfig['name'], load_if_exists=True)

    try:
        study.optimize(optimizationObjective, n_trials=jconfig['optimizer']['numTrials'], \
                       n_jobs=jconfig['optimizer']['numJobs'])
        # Use catch param?
        # Use storage='sqlite:///example.db'
    except KeyboardInterrupt:
        pass

    results = study.trials_dataframe()
    results.to_csv(join(optimizationDir, 'optimizationTrials.csv'))


# Import puppet configuration and run single/sequential tests
else:

    if 'only' in jconfig['plot']:
        # Instead of running puppet, simply make some plots
        print('=================================')
        print('===== Plot Priority Enabled =====')
        print('=== jarvis config plot = only ===')
        print('=================================')
        auxSeqTestDir2 = join(jconfig['outputDir'], plotConfig['plotOnlyParams']['dir'])
        currentPlotConf = makePlotConf(plotConfig, 'plotOnlyParams', {'plotX': plotConfig['x']})
        # Get plot possibilities for selected mode
        g = (e for e in argListPlots if e.get('name') == 'onlyPlotTypes')
        plotTypes = next(g)
        makePrettyPlots(currentPlotConf, auxSeqTestDir2, plotTypes['possibilities'], unify=False)
        exit()

    if jconfig['seq']:

        if 'confSeq' not in jconfig.keys():
            print('> Missing configuration file for sequential testing - exiting')
            exit()

        configs = getConfiguration(jconfig['confSeq'])['configs']
        origDir = getWorkDir(jconfig, 'seq - {}'.format(jconfig['name']),
                                completedText=jconfig['successString'])

        for seqConf in configs:
            pconfigs = []

            name = seqConf['name']
            variations = seqConf['variations']
            variationsWithin = seqConf['variationsWithin']
            del seqConf['variations']
            del seqConf['variationsWithin']

            variationNames = [v['name'] for v in variations]
            variationValues = {v['name']: v['values'] for v in variations}
            variationWithinNames = [v['subName'] for v in variationsWithin]
            variationWithinValues = {v['subName']: (v['name'], v['values'],) for v in variationsWithin}

            pipeline = []

            for p in seqConf['priorityLine']:
                if p in variationNames:
                    pipeline.append({'name': p, 'values': variationValues[p]})
                elif p in variationWithinNames:
                    pipeline.append({'name': variationWithinValues[p][0], 'subName': p,
                                     'values': variationWithinValues[p][1]})

            pconfigs = recursiveThingy(pipeline, seqConf, argListPuppet)

            print("==========        RUNNING TEST RUN - [{}]     ==========".format(jconfig['name']))
            # Create Directory for outputs
            seqTestDir = getWorkDir({'outputDir': origDir}, name, completedText=jconfig['successString'])

            # Todo - this should be recursive too!! right now it only allows 2 parameters to be varied
            assert(len(pipeline) == 2)    # just for now
            pipelineValues = [t['values'] for t in pipeline]

            it=0
            for var1 in pipelineValues[0]:
                auxSeqTestDir = getWorkDir({'outputDir': seqTestDir}, '{} - {}'.format(pipeline[0]['name'], var1),
                                           completedText=jconfig['successString'])
                for var2 in pipelineValues[1]:
                    pconfig = pconfigs[it]
                    it+=1

                    print("=== NEW INSTANCE ==  ")
                    printDict(pconfig, statement="> Using args:")
                    # Create output directory for instance inside sequential-test directory
                    auxSeqTestDir2 = makeDir(auxSeqTestDir, 'testrun', completedText=jconfig['successString'])

                    puppet = Puppet(pconfig, debug=jconfig['debug'], outputDir=auxSeqTestDir2)
                    puppet.pipeline()
                    dumpConfiguration(pconfig, auxSeqTestDir2, unfoldConfigWith=argListPuppet)

                    if 'single' in jconfig['plot']:
                        # Get plot possibilities for selected mode
                        currentPlotConf = makePlotConf(plotConfig, 'plotSingleParams', seqConf)
                        g = (e for e in argListPlots if e.get('name') == 'singlePlotTypes')
                        plotTypes = next(g)
                        makePrettyPlots(currentPlotConf, auxSeqTestDir2, plotTypes['possibilities'], unify=False)
                    changeDirName(auxSeqTestDir2, extraText=jconfig['successString'])

                if 'seq' in jconfig['plot']:
                    # Get plot possibilities for selected mode
                    currentPlotConf = makePlotConf(plotConfig, 'plotSeqParams', seqConf)
                    g = (e for e in argListPlots if e.get('name') == 'seqPlotTypes')
                    plotTypes = copy.deepcopy(next(g))
                    makePrettyPlots(currentPlotConf, auxSeqTestDir, plotTypes['possibilities'], unify=True,
                                    logFile='logs.json', configFile='config.yaml', unificationType=plotConfig['seqLogConversion'])
                changeDirName(auxSeqTestDir, extraText=jconfig['successString'])

            if 'seq' in jconfig['plot']:
                # Get plot possibilities for selected mode
                currentPlotConf = makePlotConf(plotConfig, 'plotSeqParams', seqConf)
                g = (e for e in argListPlots if e.get('name') == 'seqPlotTypes')
                plotTypes = copy.deepcopy(next(g))
                for k, v in seqConf['changedToHigherDimPlot'].items():
                    currentPlotConf[k] = v
                un = False

                makePrettyPlots(currentPlotConf, seqTestDir, plotTypes['possibilities'], unify=un)

            changeDirName(seqTestDir, extraText=jconfig['successString'])

        changeDirName(origDir, extraText=jconfig['successString'])

    else:
        # Single test:

        print('> Importing puppet configuration from {}'.format(jconfig['conf']))
        pconfig = getConfiguration(jconfig['conf'])
        pconfig = selectFuncAccordingToParams(pconfig, argListPuppet)

        # Upgrade arguments if command line ones were passed and attribute None value to params which were not passed
        for arg in argListPuppet:
            if arg['name'] in argsPassedPuppet:
                pconfig[arg['name']] = clArgs[arg['name']]
            elif arg['name'] not in pconfig.keys():
                pconfig[arg['name']] = None

        print("==========        RUNNING TEST RUN - [{}]     ==========".format(jconfig['name']))
        printDict(pconfig, statement="> Using args:")

        # Create output directory for instance
        auxSeqTestDir2 = getWorkDir(jconfig, jconfig['name'], completedText=jconfig['successString'])

        # Run instance
        puppet = Puppet(args=pconfig, debug=jconfig['debug'], outputDir=auxSeqTestDir2)
        puppet.pipeline()
        dumpConfiguration(pconfig, auxSeqTestDir2, unfoldConfigWith=argListPuppet)

        if 'single' in jconfig['plot']:
            # Get plot possibilities for selected mode
            currentPlotConf = makePlotConf(plotConfig, 'plotSingleParams', pconfig)
            g = (e for e in argListPlots if e.get('name') == 'singlePlotTypes')
            plotTypes = next(g)
            makePrettyPlots(currentPlotConf, auxSeqTestDir2, plotTypes['possibilities'], unify=False)

        changeDirName(auxSeqTestDir2, extraText=jconfig['successString'])


