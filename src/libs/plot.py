import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.tree import export_graphviz
import seaborn as sns
from subprocess import call
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.neighbors import NearestNeighbors

def eps_plot(data, file = None):
	nn = NearestNeighbors(n_neighbors=2)
	nbrs = nn.fit(data)
	distances, indices = nbrs.kneighbors(data)
	distances = np.sort(distances, axis = 0)
	distances = distances[:,1]
	plt.plot(distances)
	plt.xlabel('Data Points')
	plt.ylabel('Distances to Neighbors')
	if(file == None):
		plt.show()
	else:
		plt.savefig(file)

def correlation_matrix(data, name, file = None, annotTreshold = 20):
	annot = False if len(data.columns) > 20 else True
	fig = plt.figure(figsize=[25, 25])
	corr_mtx = data.corr(method='spearman')
	sns.heatmap(corr_mtx, xticklabels=False, yticklabels=False, annot=None, cmap='Blues', vmin=-1,vmax=1)
	#plt.title('Correlation analysis of {}'.format(name))
	
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

def pca_plot(data, predict, file = None, title=None):
	pca = PCA(n_components=3)
	principalComponents = pca.fit_transform(data)

	principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
	finalDf = pd.concat([principalDf, pd.DataFrame(predict)], axis=1)
	finalDf.rename(columns={0: 'class'}, inplace=True)

	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1,projection='3d') 
	ax.set_xlabel('Principal Component 1', fontsize = 15)
	ax.set_ylabel('Principal Component 2', fontsize = 15)
	ax.set_zlabel('Principal Component 3', fontsize = 15)
	ax.set_title(title, fontsize = 20)

	targets = np.arange(len(np.unique(predict)))
	for target in targets:
		indicesToKeep = finalDf['class'] == target
		ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'] , zs= finalDf.loc[indicesToKeep, 'principal component 3'] , s = 50)

	ax.legend(targets)
	ax.grid()

	if(file == None):
		plt.show()
	else:
		plt.savefig('{}/{}.png'.format(dir, filename))


def pca_plot_3d(data, predict, dir, filename = "pca3d", title=None, show = False):
	if len(data.columns) < 3:
		return 

	pca = PCA(n_components=3)
	principalComponents = pca.fit_transform(data)

	principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
	finalDf = pd.concat([principalDf, pd.DataFrame(predict)], axis=1)
	finalDf.rename(columns={0: 'class'}, inplace=True)

	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1,projection='3d') 
	ax.set_xlabel('Principal Component 1', fontsize = 15)
	ax.set_ylabel('Principal Component 2', fontsize = 15)
	ax.set_zlabel('Principal Component 3', fontsize = 15)
	ax.set_title(title, fontsize = 20)

	targets = np.arange(len(np.unique(predict)))
	for target in targets:
		indicesToKeep = finalDf['class'] == target
		ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'] , zs= finalDf.loc[indicesToKeep, 'principal component 3'] , s = 50)

	ax.legend(targets)
	ax.grid()

	if(show):
		plt.show()
	else:
		plt.savefig('{}/{}.png'.format(dir, filename))
