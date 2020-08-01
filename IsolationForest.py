import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv('train.csv')
model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(df[['target']])


df['scores']=model.decision_function(df[['target']])
df['anomaly']=model.predict(df[['target']])

anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)

outliers_counter = len(df[df['target'] > 99999])
outliers_counter
print("Accuracy percentage:", 100*list(df['anomaly']).count(-1)/(outliers_counter))
