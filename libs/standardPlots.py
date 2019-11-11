from os.path import join
from os import walk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import networkx as nx
import numpy as np
from math import sqrt, log10
from matplotlib.font_manager import FontProperties
import ast

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
reds = ['lightcoral', 'indianred', 'darkred', 'r', 'lightsalmon']
blues = ['deepskyblue', 'lightskyblue', 'darkcyan', 'steelblue', 'azure']
greens = ['g', 'limegreen', 'forestgreen', 'mediumseagrean', 'palegreen']
greys = ['dimgrey', 'darkgrey', 'lightgrey', 'slategrey', 'silver']
pinks = ['magenta', 'violet', 'purple', 'hotpink', 'pink']
colorPallets = [reds, blues, greens, greys, pinks]
colorPalletsBox = ['#2C7BB6', '#D7191C', 'grey', 'orange']


def plotThemBoxes(level, dir, x, ys, logFile, yLabelsBox=[], ymin=None, ymax=None, yAxesBox='', dpi=180):

    yLabels = yLabelsBox
    yAxes = yAxesBox
    data = fetchData(dir, level, logFile)

    fig, ax = plt.subplots()

    for i, res in enumerate(data):
        if x == None or x == "index":
            xSorted = pd.Series([str(v) for v in res.index.values])
            resSorted = res
        else:
            resSorted = res.sort_values(by=[x])
            xSorted = resSorted[x]

        for j, y in enumerate(ys):
            ySorted = resSorted[y]  # List of values correspondent correspondent to the fixed one
            ySorted = [ast.literal_eval(ySorted.values[j]) for j in range(len(ySorted))]     # convert string to list ( each one with several kfold values)

            max = xSorted.max() if type(xSorted[0]) is not str else len(xSorted)
            numDataPoints = len(xSorted)
            boxesPerDataPoint = len(data)
            distBetweenDataPoints = max/numDataPoints
            deltaValue = distBetweenDataPoints/boxesPerDataPoint
            width = deltaValue/3

            if len(data) == 1:
                delta = [0]
            if len(data) == 2:
                delta = [-deltaValue, deltaValue]
            elif len(data) == 3:
                delta = [-deltaValue, 0, deltaValue]

            # Move plots to the sides
            xValues = xSorted if type(xSorted[0]) is not str else list(range(len(xSorted)))
            dislocatedX = [z+delta[i] for z in xValues] if delta else xValues

            # Create plot
            # width = 1     # 0.1
            bpl = plt.boxplot(ySorted, positions=dislocatedX, sym='', widths=width)
            set_box_color(bpl, colorPalletsBox[i])

    # Associate label and color
    for i, lab in enumerate(yLabels):
        plt.plot([], c=colorPalletsBox[i], label=lab)
    plt.legend()

    plt.xlabel(x)
    if yAxes:
        plt.ylabel(yAxes)

    # X axis:
    xAxis = [str(j) for j in xSorted]
    plt.xticks(xValues, xAxis)

    # Increase x minimum to catch a bit more on the left
    xmin = xValues[0]-np.diff(xValues)[0]
    ax.set_xlim(left=xmin)

    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax + ymax * 0.1)
    ax.legend()

    plt.tight_layout()
    outputName = buildOutputName(x, ys, dir)
    plt.savefig(outputName + ' - box.png', dpi=dpi)
    plt.close(fig=fig)

# Only supports logfile CSV
def plotDemStats(level, dir, x, ys, logFile, yLabelsLine=[], yAxes='', ymin=None, ymax=None, pallets=False, dpi=180,
                 joinYToLabel=None):

    yLabels = yLabelsLine
    data = fetchData(dir, level, logFile)

    fig, ax = plt.subplots()

    # labelsToUse = yLabels if len(yLabels) == len(data) else ['' for i in data]
    for i, res in enumerate(data):
        shape = len(data) if len(data)==len(yLabels) else data[0].shape[0]
        if len(yLabels) == shape:
            # takes priority
            labels = [yLabels[i] for j,_ in enumerate(ys)]
        elif len(yLabels) == len(ys):
            labels = yLabels
        else:
            labels = ['' for y in yLabels]

        if pallets:
            multipleYsLinePlot(ax, res, ys, x, colors=colorPallets[i], labels=labels, joinYToLabel=joinYToLabel)
        else:
            multipleYsLinePlot(ax, res, ys, x, labels=labels, joinYToLabel=joinYToLabel)

    plt.xlabel(x)
    if yAxes:
        plt.ylabel(yAxes)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax + ymax * 0.1)
    ax.legend()
    outputName = buildOutputName(x, ys, dir)
    plt.savefig(outputName + ' - linePlot.png', dpi=dpi)
    plt.close(fig=fig)


def multipleYsLinePlot(ax, data, y_types, x_type, colors=[], labels=[], joinYToLabel=None):
    '''
    :param data:    (pd.Dataframe) Data out of output.csv
    :param y_types: (array) Headers to be used from output.csv
    :param x_type:  (str) Single header to be used as x, from output.csv
    :return:
    '''
    
    assert(len(y_types)>0)
    if x_type == None or x_type == "index":
        x = [str(v) for v in data.index.values]
    else:
        data = data.sort_values(by=[x_type])
        x = data[x_type]

    maxNum = 0
    labelsToUse = labels if len(labels) == len(y_types) else ['' for y in y_types]   # fixme - labels[0]
    for i, y in enumerate(y_types):
        l = labelsToUse[i] + ' - {}'.format(y) if joinYToLabel else labelsToUse[i]
        if colors:
            ax.plot(x, data[y], label=l, color=colors[i])
        else:
            ax.plot(x, data[y], label=l)
        maxNum = max(data[y]) if max(data[y]) > maxNum else maxNum


def scaterThemPlotsNx(level, dir, x, ys, logFile, dpi, yLabelsScatter=None, ymin=None, ymax=None, annotations=False, distanceToLabel=None):

    data = fetchData(dir, level, logFile)

    fig, ax = plt.subplots()
    G = nx.Graph()

    data_nodes, init_pos = [], {}
    nodes, labels, bestBoys, secondBestBoys = [], [], [], []
    it = 0
    for i, res in enumerate(data):
        assert (len(ys) > 0)
        if x == None or x == "index":
            xVal = [str(v) for v in res.index.values]
            resSorted = res
        else:
            resSorted = res.sort_values(by=[x])
            xVal = resSorted[x]


        x_ = resSorted[ys[0]]
        y_ = resSorted[ys[1]]
        yLabels = yLabelsScatter if len(yLabelsScatter) == len(data) else [[''] for j in y_ for i in data]
        ax.scatter(x_, y_, label=yLabels[i])
        dists = []
        for x__, y__ in zip(x_, y_):
            d = distance((x__, y__), (0, 0))
            dists.append(d)

        top3Dists = sorted(dists, reverse=True)[:3]
        other2Dists = sorted(dists, reverse=True)[3:5]
        for j, p in enumerate(x_):
            nodeStr = 'node-{}'.format(it)
            if dists[j] in top3Dists:
                bestBoys.append(it)
            if dists[j] in other2Dists:
                secondBestBoys.append(it)
            it+=1
            node = (p, resSorted[ys[1]][j])
            annotation = xVal[j]
            nodes.append(node)
            labels.append(annotation)
            G.add_node(nodeStr)
            G.add_node(annotation)
            G.add_edge(nodeStr, annotation)
            data_nodes.append(nodeStr)
            init_pos[nodeStr] = node
            init_pos[annotation] = node

    pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=distanceToLabel)

    for j in range(it):
        data_str = 'node-{}'.format(j)
        ano_str = labels[j]

        if j in bestBoys or j in secondBestBoys:
            if j in bestBoys:
                color='red'
            else:
                color= 'lightcoral'
        else:
            color = 'black'
        aph = 0.3 if j not in bestBoys and bestBoys else 0.6
        ax.annotate(ano_str,
                    xy=pos[data_str], xycoords='data',
                    xytext=pos[ano_str], textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    alpha=aph,
                                    color=color,
                                    connectionstyle="arc3"))

    plt.xlabel(ys[0])
    plt.ylabel(ys[1])

    ax.set_xlim(ymin, ymax)
    ax.set_ylim(ymin, ymax)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    outputName = buildOutputName(x, ys, dir)
    # plt.show()
    plt.savefig(outputName + ' - scatterNx.png', dpi=dpi)
    plt.close(fig=fig)



def scaterThemPlots(level, dir, x, ys, logFile, dpi, yLabelsScatter, ymin=None, ymax=None, annotations=False):



    data = fetchData(dir, level, logFile)
    if level == 'localCSV' and len(yLabelsScatter) == data[0].shape[0]:
        aux = list(data[0].T.to_dict().values())
        data = [pd.DataFrame(p, index=[i for i in range(len(p.keys()))]) for p in aux]
    '''
    if len(data==1) and len(yLabels) == data[0].shape[0]:
        transform data into list of arrays.
        
    shape = len(data) if len(data) == len(yLabelsScatter) else data[0].shape[0]
    if len(yLabelsScatter) == shape:
        # takes priority
        labels = [yLabelsScatter[i] for i in range(shape)]
    '''
    yLabels = yLabelsScatter if len(yLabelsScatter)== len(data) else ['' for i in data]

    fig, ax = plt.subplots()

    for i, res in enumerate(data):
        if annotations:
            assert(len(ys)>0)
            if x == None or x == "index":
                xVal = [str(v) for v in res.index.values]
                resSorted = res
            else:
                resSorted = res.sort_values(by=[x])
                xVal = resSorted[x]

            ax.scatter(resSorted[ys[0]], resSorted[ys[1]], label=yLabels[i])
        else:
            ax.scatter(res[ys[0]], res[ys[1]], label=yLabels[i][0])

        '''
        if annotations:
            for i, txt in enumerate(xVal):
                ax.annotate(txt, (resSorted[ys[0]][i], resSorted[ys[1]][i]))
        '''
    plt.xlabel(ys[0])
    plt.ylabel(ys[1])

    # fixme - cheating labbels because adding more variables is boring
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
        ax.set_xlim(left=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax + ymax * 0.1)
        ax.set_xlim(right=ymax + ymax * 0.1)
    ax.legend()

    outputName = buildOutputName(x, ys, dir)
    plt.savefig(outputName + ' - scatter.png', dpi=dpi)
    plt.close(fig=fig)



def fetchData(dir, level, logFile):
    if level == 'localCSV':
        csvLocation = join(dir, logFile)
        data = pd.read_csv(csvLocation, sep='\t', index_col=0, encoding='utf-8')
        data = [data]
    elif level == 'fetchCSV':
        data = fetchLogsFromDirs(logFile, dir)
    else:
        print('unkown level')
        exit()
    return data


def fetchLogsFromDirs(logFile, dir):
    results = []
    for subdir, dirs, files in walk(dir):
        for f in dirs:
            outFile = join(subdir, f, logFile)
            data = pd.read_csv(outFile, sep='\t', index_col=0, encoding='utf-8')
            results.append(data)
        break  # Only apply recursivness once
    return results


def buildOutputName(x, ys, dir):
    outputName = x + ' by ['
    for n in ys:
        outputName += '{}, '.format(n)
    outputName = outputName + ']'
    outputName = join(dir, outputName)
    return outputName


def distance(p0, p1):
    return sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)