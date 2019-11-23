import pandas as pd
from libs.treatment import *
from sklearn.feature_selection import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, minmax_scale


sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

def count_nan_vars(df):
	nan_vars = 0
	for el in df:
		if df[el].isna().sum() > 0:
			nan_vars += 1

	return nan_vars

def count_type_vars(dtypes):
	count = { }
	for el in dtypes:
		if el in count:
			count[el] += 1
		else:
			count[el] = 1
	return count


def do_single_variable_analysis(filename, header = 0, k = 10, max_corr = 0.94):
	df = pd.read_csv(filename, header = header, sep=',', decimal='.')
	
	"""data_types = count_type_vars(df.dtypes)	
	nan_vars = count_nan_vars(df)
	print(data_types)
	print(data_types)	
	print("Nan count {}".format(nan_vars))
	print("shape {}".format(df.shape))
	print("size {}".format(df.size))
	print("Class count {}".format((df.iloc[:,-1].value_counts())))
	print(df.describe())
	"""
	categorical_cols = [0, 1]
	x = df.iloc[:, :-1]
	y = df.iloc[:,-1]

	x, _ = dropHighCorrFeat(x, max_corr = max_corr)
	print('x Train state: {}'.format(x.shape))
	columns = SelectKBest(f_classif, k = k).fit(x, y).get_support()
	new_x = x.loc[:,columns]
	new_x = normalize(new_x)
	new_x = new_x.stack().reset_index().rename(columns={0:'x', 'level_1': 'g'}).drop(columns=['level_0'])
	plot_dist(new_x, k,max_corr)
	print(list(new_x))

def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


def plot_dist(df, k, max_corr):
	pal = sns.cubehelix_palette(10, rot=-.1, light=.4)
	g = sns.FacetGrid(df, row="g", hue="g", aspect=20, height=.5, palette=pal)
	g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
	g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
	g.map(plt.axhline, y=0, lw=2, clip_on=False)
	g.map(label, "x")
	g.fig.subplots_adjust(hspace=-.25)
	g.set_titles("")
	g.set(yticks=[])
	g.despine(bottom=True, left=True)
	plt.savefig("dist_{}_{}.png".format(k, max_corr))

do_single_variable_analysis('./data/pd_speech_features.csv', header = 1)
#do_single_variable_analysis('./data/covtype.data', header = None)