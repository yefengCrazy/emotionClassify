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

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

# 获取数据 [1995 rows x 400 columns]
fdir = ''
df = pd.read_csv(fdir + '2000_data.csv')
y = df.iloc[:,1]
x = df.iloc[:,2:]

##根据图形取100维
# x_pca = PCA(n_components = 2).fit_transform(x)
x_pca=x
#划分数据集
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(
    x_pca,y,random_state=1,train_size=0.1
)

def plt_subplot(x_train,y_train,path,title):
    temp_1 = []
    temp_0 = []
    for i in range(len(x_train.tolist())):
        if y_train.tolist()[i] == 1.0:
            temp_1.append(x_train[i])
        else:
            temp_0.append(x_train[i])

    plt.figure(figsize=(8, 6))

    plt.scatter(np.array(temp_0)[:, 0], np.array(temp_0)[:, 1], marker='o',
                cmap='Reds',
                c='#FF7F0E',
                edgecolor='#FF7F0E', alpha=0.5, label='Preferential praise type')
    plt.scatter(np.array(temp_1)[:, 0], np.array(temp_1)[:, 1], marker='o',
                cmap='Reds',
                c='#1F77B4',
                edgecolor='#1F77B4', alpha=0.5, label='Partial conservative type')
    plt.grid(True, linestyle='--')
    plt.title(title)
    plt.ylabel('Variable X')
    plt.xlabel('Variable Y')
    plt.legend(loc='upper right')
    plt.savefig(path)
    plt.show()

def dataset_show(x_test,y_test):

    #https://blog.csdn.net/Together_CZ/article/details/80002373
    '''
    数据及样本可视化
    原始样本数据经过采样后会形成将近1:1的数据集
    '''
    X= x_test
    y=y_test
    print('Original dataset shape {}'.format(Counter(y)))
    startnum=221
    pca=PCA(n_components=2)
    #将数据矩阵转化为二维矩阵
    X_vis=pca.fit_transform(X)
    #应用SMOTE
    sm=SMOTE()
    X_resampled, y_resampled_SMOTE=sm.fit_sample(X, y)
    print('SMOTE dataset shape {}'.format(Counter(y_resampled_SMOTE)))
    X_res_vis_SMOTE=pca.transform(X_resampled)
    #应用SMOTE + ENN
    sm=SMOTEENN()
    X_resampled, y_resampled_SMOTEENN=sm.fit_sample(X, y)
    print('SMOTEENN dataset shape {}'.format(Counter(y_resampled_SMOTEENN)))
    X_res_vis_SMOTEENN=pca.transform(X_resampled)
    #应用SMOTE + Tomek
    smt=SMOTETomek()
    X_resampled, y_resampled_SMOTETomek=smt.fit_sample(X, y)
    print('SMOTETomek dataset shape {}'.format(Counter(y_resampled_SMOTETomek)))
    X_res_vis_SMOTETomek=pca.transform(X_resampled)
    # plot_data_list=[[X_vis,y],[X_res_vis_SMOTE,y_resampled_SMOTE],[X_res_vis_SMOTEENN,y_resampled_SMOTEENN],[X_res_vis_SMOTETomek,y_resampled_SMOTETomek]]
    # plt.subplot(2,2,1)
    # plt.plot(plot_data_list[0][0],plot_data_list[0][1])
    # plt.show()
    plt_subplot(X_vis,y, 'chushi.png', 'chushi')
    plt_subplot(X_res_vis_SMOTE,y_resampled_SMOTE,'SMOTE.png','SMOTE')
    plt_subplot(X_res_vis_SMOTEENN,y_resampled_SMOTEENN, 'SMOTEENN.png', 'SMOTEENN')
    plt_subplot(X_res_vis_SMOTETomek,y_resampled_SMOTETomek, 'SMOTETomek.png', 'SMOTETomek')

dataset_show(x_test,y_test)


