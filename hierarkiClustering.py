# NAMA  : DIKI FAKHRIZAL
# NIM   : 202251101
# KELAS : Praktikum Pengenalan Pola - D 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly as py
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#print('tes')

df = pd.read_csv('Mall_Customers.csv')
df.describe()
plt.figure(1, figsize = (15,6))
n = 0

for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    sns.distplot(df[x], bins=15)
    plt.title('Distplot of {}'.format(x))

plt.show()

# Label Encoding
# Coding label mengacu pada mengubah label menjadi bentuk numerik
# sehingga, mengubahnya menjadi bentuk yang dapat dibaca oleh mesin komputer
# Algoritma pembelajaran mesin kemudian dapat memutuskan dengan baik
# bagaimana label tersebut terus dioperasikan

label_encoder = preprocessing.LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])
df.head()

# Dendrogram
plt.figure(1, figsize = (16,8))
dendrogram = sch.dendrogram(sch.linkage(df, method = "ward"))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()


# Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'average')

y_hc = hc.fit_predict(df)
print(y_hc)

# Plot Agglomerative Clustering
x = df.iloc[:, [3,4]].values
plt.scatter(x[y_hc==0, 0], x[y_hc==0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_hc==1, 0], x[y_hc==1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_hc==2, 0], x[y_hc==2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y_hc==3, 0], x[y_hc==3, 1], s=100, c='purple', label='Cluster 4')
plt.scatter(x[y_hc==4, 0], x[y_hc==4, 1], s=100, c='orange', label='Cluster 5')

plt.title('Cluster of Customers (Hierarchical Clustering Model)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.show()


