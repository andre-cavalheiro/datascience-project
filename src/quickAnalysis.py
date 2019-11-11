import pandas as pd
from libs.treatment import *

df = pd.read_csv('data/pd_speech_features.csv', header=1, sep=',', decimal='.')
print(df.head())
k=df.describe()
print(k)