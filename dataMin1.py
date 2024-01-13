# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:15:02 2023

@author: vaish
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#now import file from data set and create a dataframe

univ1 = pd.read_excel("C:/2-dataset/University_Clustering.xlsx")

a = univ1.describe()
#we have one column 'state' which is really not useful we will drop it 

univ = univ1.drop(['State'],axis = 1)
'''
we know that there is scale difference among the columns either 
by using normalization or standardization 
whenever there is mixed data apply normalization 
'''

def norm_func(i):
    x = (i-i.min()) / (i.max() - i.min())
    return x

'''
now apply this normalization function to univ dataframe
for all the rows and column from 1 until end
since 0 the column has university name here hence skipped 
'''

df_norm = norm_func(univ.iloc[:,1:])

'''
u can check the df_norm dataframe which is scaled
between values from 0 to 1
u can apply describe function to new data frame
'''

b = df_norm.describe()

'''
before u can apply clustering u need to plot dendrogram
now to create dendrogram we need to measure distance
ref help for the linkage
'''

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

'''
linkage function gives us hierarchical  or aglomerative clustering 
ref the help for linkage
'''

z  = linkage(df_norm , method = 'complete' , metric='euclidean')
plt.figure(figsize=(15,8));
plt.title('Hierarchical Clustering dendrogram');
plt.xlabel("Index");
plt.ylabel("Distance");

'''
ref help of dendrogram 
sch.dendrogram(z)
'''

sch.dendrogram(z,leaf_rotation = 0 , leaf_font_size = 10)
plt.show()

'''
dentrigram()
applying aggilmerative clustering  choosing 3 as cluster
whatever has been displayed in dendrogram is not clustering 
it is just showing number of possible cluster
'''

from sklearn.cluster import  AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='euclidean').fit(df_norm)

#apply lables to the  cluster 

h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to univ dataframe as column and name the column

univ['clust'] = cluster_labels
#we want to relocate the column 7 and 0th position

univ1.iloc[:,2:].groupby(univ1.clust).mean()

'''
from the output cluster 2 has got highest Top10
lowest accept ratio best faculty ratio and highest expenses 
highest graduate ratio 
'''

univ1.to_csv("University.csv",encoding = 'utf-8')
import os
os.getcwd()
