#!/usr/bin/env python
# -*- coding: utf-8  -*-
# PCA  SVM
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
import sklearn.model_selection
from imblearn.under_sampling import RandomUnderSampler


# 获取数据 [1995 rows x 400 columns]
fdir = ''
df = pd.read_csv(fdir + '2000_data.csv')
y = df.iloc[:,1]
x = df.iloc[:,2:]

##根据图形取100维
x_pca = PCA(n_components = 2).fit_transform(x)

#划分数据集
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(
    x_pca,y,random_state=1,train_size=0.7
)

temp_1=[]
temp_0=[]
for i in range(len(x_train.tolist())):
    if y_train.tolist()[i]==1.0:
        temp_1.append(x_train[i])
    else:
        temp_0.append(x_train[i])


plt.figure(figsize=(8,6))

plt.scatter(np.array(temp_0)[:,0],np.array(temp_0)[:,1],marker='o',
            cmap='Reds',
            c='#FF7F0E',
            edgecolor='#FF7F0E',alpha=0.5,label='Preferential praise type')
plt.scatter(np.array(temp_1)[:,0],np.array(temp_1)[:,1],marker='o',
            cmap='Reds',
            c='#1F77B4',
            edgecolor='#1F77B4',alpha=0.5,label='Partial conservative type')
plt.grid(True, linestyle='--')
plt.title('123')
plt.ylabel('variable X')
plt.xlabel('Variable Y')
plt.legend(loc='upper right')
plt.savefig('test.png')
plt.show()









