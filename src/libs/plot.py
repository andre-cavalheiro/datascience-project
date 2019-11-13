import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.tree import export_graphviz
import seaborn as sns
from subprocess import call

def correlation_matrix(data, name, file = None, annotTreshold = 20):
	annot = False if len(data.columns) > 20 else True

	fig = plt.figure(figsize=[25, 25])
	corr_mtx = data.corr(method='spearman')
	sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=annot, cmap='Blues')
	plt.title('Correlation analysis of {}'.format(name))
	
	if(file == None):
		plt.show()
	else:
		plt.savefig(file)
	plt.close(fig=fig)

def sparsity(data, file = None):
	columns = data.select_dtypes(include='number').columns
	rows, cols = len(columns)-1, len(columns)-1
	plt.figure()
	fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
	for i in range(len(columns)):
	    var1 = columns[i]
	    for j in range(i+1, len(columns)):
	        var2 = columns[j]
	        axs[i, j-1].set_title("%s x %s"%(var1,var2))
	        axs[i, j-1].set_xlabel(var1)
	        axs[i, j-1].set_ylabel(var2)
	        axs[i, j-1].scatter(data[var1], data[var2])
	fig.tight_layout()

	if(file == None):
		plt.show()
	else:
		plt.savefig(file)
	plt.close(fig=fig)

def sens_spec_scatter(inputF, file=None, name="Sensitivity and Sensitivity", sensLabel = 'sensitivity', \
						specLabel = 'specificity', label = 'balancingStrategy'):
	data = pd.read_csv(inputF,  sep=',', encoding='utf-8')
	fig, ax = plt.subplots()

	data = pd.read_csv("../data/pd_speech_features.csv", header=[0,1])
	plt.xlabel('Sensitivity')
	plt.ylabel('Specificity')

	for index, row in data.iterrows():
		ax.scatter(row[sensLabel], row[specLabel], label=row[label], edgecolors='none')

	plt.title('Sensitivity and specificity of {}'.format(name))
	ax.legend()
	ax.grid(True)

	if(file == None):
		plt.show()
	else:
		plt.savefig(file)

def decision_tree_visualizer(tree, dir, filename = "dtree", show = False):
	dot_file = '{}/{}.dot'.format(dir, filename)
	png_file = '{}/{}.png'.format(dir, filename)

	dot_data = export_graphviz(tree, out_file=dot_file, filled=True, rounded=True, special_characters=True)  
	call(['dot', '-Tpng', dot_file, '-o', png_file, '-Gdpi=600'])

	if show:
		plt.figure(figsize = (14, 18))
		plt.imshow(plt.imread(png_file))
		plt.axis('off')
		plt.show()
