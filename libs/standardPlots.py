from os.path import join
from os import walk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import networkx as nx
import numpy as np
from math import sqrt
from matplotlib.font_manager import FontProperties

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
reds = ['lightcoral', 'indianred', 'darkred', 'r', 'lightsalmon']
blues = ['deepskyblue', 'darkcyan', 'lightskyblue', 'steelblue', 'azure']
greens = ['g', 'limegreen', 'forestgreen', 'mediumseagrean', 'palegreen']
greys = ['dimgrey', 'darkgrey', 'lightgrey', 'slategrey', 'silver']
pinks = ['magenta', 'violet', 'purple', 'hotpink', 'pink']
colorPallets = [reds, blues, greens, greys, pinks]

"""
# Only supports logfile CSV
def plotDemStats(dir, x, ys, logFile, yAxes=[], ymin=None, ymax=None, dpi=180):
    outputName = x + ' by ['
    for n in ys:
        outputName += '{}, '.format(n)
    outputName = outputName + ']'
    outputName = join(dir, outputName)

    csvLocation = join(dir, logFile)
    data = pd.read_csv(csvLocation, sep='\t', index_col=0, encoding='utf-8')
    '''fig, ax = plt.subplots()
    multipleYsLinePlot(ax, data, ys, x, colors=[], dpi=180)
    plt.xlabel(x)
    if yAxes:
        plt.ylabel(yAxes)
    if ymin:
        ax.set_ylim(bottom=0)
    if ymax:
        ax.set_ylim(top=ymax + ymax * 0.1)
    ax.legend()
    plt.savefig(outputName + '.png', dpi=dpi)'''
    makeImage([data], ys, x, outputName, yAxes, ymin, ymax, dpi)
"""

# Only supports logfile CSV
def plotDemStats(level, dir, x, ys, logFile, yLabels=[], yAxes='', ymin=None, ymax=None, pallets=False, dpi=180,
                 joinYToLabel=None):


    data = fetchData(dir, level, logFile)

    fig, ax = plt.subplots()

    # labelsToUse = yLabels if len(yLabels) == len(data) else ['' for i in data]
    for i, res in enumerate(data):
        if pallets:
            multipleYsLinePlot(ax, res, ys, x, colors=colorPallets[i], labels=yLabels[i], joinYToLabel=joinYToLabel)
        else:
            multipleYsLinePlot(ax, res, ys, x, labels=yLabels[i], joinYToLabel=joinYToLabel)

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
    labelsToUse = labels if len(labels) == len(y_types) else [labels[0] for y in y_types]   # fixme - labels[0]
    for i, y in enumerate(y_types):
        l = labelsToUse[i] + ' - {}'.format(y) if joinYToLabel else labelsToUse[i]
        if colors:
            ax.plot(x, data[y], label=l, color=colors[i])
        else:
            ax.plot(x, data[y], label=l)
        maxNum = max(data[y]) if max(data[y]) > maxNum else maxNum


def scaterThemPlotsNx(level, dir, x, ys, logFile, dpi, yLabels=None, ymin=None, ymax=None, annotations=False, distanceToLabel=None):

    data = fetchData(dir, level, logFile)

    fig, ax = plt.subplots()
    G = nx.Graph()

    data_nodes, init_pos = [], {}
    nodes, labels, bestBoys, secondBestBoys = [], [], [], []
    it = 0
    for i, res in enumerate(data):
        assert (len(ys) > 0)
        resSorted = res.sort_values(by=[x])
        xVal = resSorted[x]

        x_ = resSorted[ys[0]]
        y_ = resSorted[ys[1]]
        ax.scatter(x_, y_, label=yLabels[i][0])
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

    all_pos = np.vstack(pos.values())

    plt.xlabel(ys[0])
    plt.ylabel(ys[1])

    ax.set_xlim(ymin, ymax)
    ax.set_ylim(ymin, ymax)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    outputName = buildOutputName(x, ys, dir)
    plt.savefig(outputName + ' - scatterNx.png', dpi=dpi)


def scaterThemPlots(level, dir, x, ys, logFile, dpi, yLabels, ymin=None, ymax=None, annotations=False):

    data = fetchData(dir, level, logFile)

    fig, ax = plt.subplots()

    for i, res in enumerate(data):
        if annotations:
            assert(len(ys)>0)
            resSorted = res.sort_values(by=[x])
            xVal = resSorted[x]

            ax.scatter(resSorted[ys[0]], resSorted[ys[1]], label=yLabels[i][0])
        else:
            ax.scatter(res[ys[0]], res[ys[1]], label=yLabels[i][0])

        if annotations:
            for i, txt in enumerate(xVal):
                ax.annotate(txt, (resSorted[ys[0]][i], resSorted[ys[1]][i]))

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