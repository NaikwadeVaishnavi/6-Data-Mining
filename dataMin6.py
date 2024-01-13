# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:17:30 2023

@author: vaish
"""

import pandas as pd
import numpy as np
from numpy import array
from scipy.linalg import svd
A = array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
print(A)

#SVD
U,d,Vt = svd(A)

print(U)
print(d)
print(Vt)
print(np.diag(d)) 

#SVD applying to dataset

import pandas as pd 

data = pd.read_excel("C:/2-dataset/University_Clustering.xlsx")
data.head()
data.describe()

data = data.iloc[:,2:] #removing the non numeriacal data
data

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=3)
svd.fit(data)

result = pd.DataFrame(svd.transform(data))
result.head()

result.columns = 'pc0','pc1','pc2'
result.head()
