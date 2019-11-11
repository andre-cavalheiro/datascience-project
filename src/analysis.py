import pandas as pd

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

def do_single_variable_analysis(filename, header = 0):
	df = pd.read_csv(filename, header = header, sep=',', decimal='.')
	
	data_types = count_type_vars(df.dtypes)	
	nan_vars = count_nan_vars(df)
	
	print(data_types)	
	print("Nan count {}".format(nan_vars))
	print("shape {}".format(df.shape))
	print("size {}".format(df.size))
	print("Class count {}".format((df.iloc[:,-1].value_counts())))
	print(df.describe())

do_single_variable_analysis('./data/data/proj/pd_speech_features.csv', header = 1)
do_single_variable_analysis('./data/data/proj/covtype.data', header = None)