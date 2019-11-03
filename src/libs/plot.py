import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

