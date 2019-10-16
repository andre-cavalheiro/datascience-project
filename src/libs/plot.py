import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import seaborn as sns

def correlation_matrix(data, name, file = None, annotTreshold = 20):
	annot = False if len(data.columns) > 20 else True

	fig = plt.figure(figsize=[15, 15])
	corr_mtx = data.corr()
	sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=annot, cmap='Blues')
	plt.title('Correlation analysis of {}'.format(name))
	
	if(file == None):
		plt.show()
	else:
		plt.savefig(file)


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


def sens_spec_scatter(inputF, file=None, name="Sensitivity and Sensitivity", sensLabel = 'sensitivity', \
						specLabel = 'specificity', label = 'balancingStrategy'):
	data = pd.read_csv(inputF,  sep='\t', encoding='utf-8')
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

sens_spec_scatter("../output/balancing with naiveBays entire dataset/output.csv",\
	file = "../output/balancing with naiveBays entire dataset/sens_spec_scatter.png", \
	name = "Naive Bays with entire dataset")
sens_spec_scatter("../output/balancing with nB entire dataset corr=.80/output.csv",\
	file = "../output/balancing with nB entire dataset corr=.80/sens_spec_scatter.png", \
	name = "Naive Bays with correlation threshold of .80")


"""
data = pd.read_csv("../data/data/proj/pd_speech_features.csv", header=[0,1])

visited = []
for column in data:
	if column[0] not in visited:
		print(column[0])
		correlation_matrix(data[column[0]],column[0], "../output/lab2/{}.jpg".format(column[0])) 
		visited.append(column[0])


print(data.head())
curr_col_name = "Special Features"
curr_df = data.iloc[:, [0]]
features_dict = []


for column in data:
	if "Unnamed" in column:
		curr_df = pd.concat([curr_df, data[column]], axis=1, join='inner')
	else:
		new_header = curr_df.iloc[0] 
		curr_df = curr_df[1:] 
		curr_df.columns = new_header  

		features_dict.append((column, curr_df))
		curr_df = data[column]


for f in features_dict[1:]:
	correlation_matrix(f[1],f[0],"{}.jpg".format(f[0])) 
"""