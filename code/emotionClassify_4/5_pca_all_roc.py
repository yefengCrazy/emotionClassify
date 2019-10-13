#!/usr/bin/env python
# -*- coding: utf-8  -*-
# PCA  SVM
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
import sklearn.model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB as GNB
import time
# 获取数据 [1995 rows x 400 columns]
fdir = ''
df = pd.read_csv(fdir + '2000_data.csv')
y = df.iloc[:,1]
x = df.iloc[:,2:]

##根据图形取100维
x_pca = PCA(n_components = 100).fit_transform(x)

#划分数据集
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(
    x_pca,y,random_state=1,train_size=0.7
)

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

print(Counter(y_train.tolist()))
print(Counter(y_test.tolist()))

# #应用SMOTE + ENN  训练集
# sme = SMOTEENN(random_state=0)
# X_res, y_res = sme.fit_resample(x_train, y_train)
# print('2',Counter(y_res))
#
# print(y_res)

from imblearn.over_sampling import RandomOverSampler as  ros
#https://blog.csdn.net/weixin_44871660/article/details/90600522
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
ros = SMOTETomek(random_state=0)
X_resample,y_resample = ros.fit_resample(x_train,y_train)
print('2',Counter(y_resample))
# sys.exit(0)
x_train=X_resample
y_train=y_resample


# SVM (RBF)
# using training data with 100 dimensions
# 支持向量机，对于特征含义相似的中等大小的数据集很强大，需要数据缩放，对参数敏感
clf = svm.SVC(C = 2, probability = True)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
#https://blog.csdn.net/hfutdog/article/details/88085878
print('准确率: ',accuracy_score(y_test,y_pred))#准确率，准确率是分类正确的样本占总样本个数的比例
print('精确率: ',precision_score(y_test, y_pred))#精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
print('召回率: ',recall_score(y_test, y_pred))#召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例
print("F1值: %.3f"%f1_score(y_test, y_pred))

pred_probas = clf.predict_proba(x_pca)[:,1] #score
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, color='c',label = 'SVM area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')

# 朴素贝叶斯，只适用于分类问题，比线性模型速度还快，适用于非常大的数据集和高维数据，但精度通常低于线性模型
gnb=GNB()
gnb.fit(X_resample,y_resample)
y_pred=gnb.predict(x_test)
print('Training Score: ' , gnb.score(x_train,y_train))
print('Testing Score: ' , gnb.score(x_test, y_test))
print(len(y_test),len(y_pred))
print('准确率: ',accuracy_score(y_test,y_pred))#准确率，准确率是分类正确的样本占总样本个数的比例
print('精确率: ',precision_score(y_test, y_pred))#精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
print('召回率: ',recall_score(y_test, y_pred))#召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例
print("F1值: %.3f"%f1_score(y_test, y_pred))

pred_probas = gnb.predict_proba(x_pca)[:,1] #score
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'Bays area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
classifier=Pipeline([('sc',StandardScaler()),('clf',LogisticRegression())])
classifier.fit(x_train,y_train.ravel())

y_pred=classifier.predict(x_test)
print('Training Score: ' , classifier.score(x_train,y_train))
print('Testing Score: ' , classifier.score(x_test, y_test))
print(len(y_test),len(y_pred))
print('准确率: ',accuracy_score(y_test,y_pred))#准确率，准确率是分类正确的样本占总样本个数的比例
print('精确率: ',precision_score(y_test, y_pred))#精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
print('召回率: ',recall_score(y_test, y_pred))#召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例
print("F1值: %.3f"%f1_score(y_test, y_pred))

pred_probas = classifier.predict_proba(x_pca)[:,1] #score
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'Logistic area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')


from sklearn.tree import DecisionTreeClassifier
# 决策树，速度很快，不需要数据缩放，可以可视化，很容易解释
dc=DecisionTreeClassifier()
dc.fit(x_train,y_train)

y_pred = dc.predict(x_test)
print('Training Score: ' , dc.score(x_train,y_train))
print('Testing Score: ' , dc.score(x_test, y_test))
print(len(y_test),len(y_pred))
print('准确率: ',accuracy_score(y_test,y_pred))#准确率，准确率是分类正确的样本占总样本个数的比例
print('精确率: ',precision_score(y_test, y_pred))#精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
print('召回率: ',recall_score(y_test, y_pred))#召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例
print("F1值: %.3f"%f1_score(y_test, y_pred))

pred_probas = dc.predict_proba(x_pca)[:,1] #score
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'DecisionTree area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')

## 梯度提升决策树，精度通常比随机森林略高，与随机森林相比，训练速度更慢，但是预测速度更快，需要的内存也少，比随机森林需要更多的参数调节
from sklearn.ensemble import GradientBoostingClassifier
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(x_train,y_train)

y_pred = gbm0.predict(x_test)
print('Training Score: ' , gbm0.score(x_train,y_train))
print('Testing Score: ' , gbm0.score(x_test, y_test))
# print(len(y_test),len(y_pred))
print('准确率: ',accuracy_score(y_test,y_pred))#准确率，准确率是分类正确的样本占总样本个数的比例
print('精确率: ',precision_score(y_test, y_pred))#精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
print('召回率: ',recall_score(y_test, y_pred))#召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例
print("F1值: %.3f"%f1_score(y_test, y_pred))

pred_probas = gbm0.predict_proba(x_pca)[:,1] #score
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'GradientBoosting area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')


# 随机森林，几乎总是比单个决策树的表现要好，鲁棒性很好，非常强大。不需要数据缩放，不适用于高维稀疏矩阵
from sklearn.ensemble import RandomForestClassifier
gbm0=RandomForestClassifier()
gbm0.fit(x_train,y_train)

y_pred = gbm0.predict(x_test)
print('Training Score: ' , gbm0.score(x_train,y_train))
print('Testing Score: ' , gbm0.score(x_test, y_test))
# print(len(y_test),len(y_pred))
print('准确率: ',accuracy_score(y_test,y_pred))#准确率，准确率是分类正确的样本占总样本个数的比例
print('精确率: ',precision_score(y_test, y_pred))#精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
print('召回率: ',recall_score(y_test, y_pred))#召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例
print("F1值: %.3f"%f1_score(y_test, y_pred))

pred_probas = gbm0.predict_proba(x_pca)[:,1] #score
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'RandomForest area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')


# 神经网络，可以构建非常复杂的模型，特别是对于大型数据而言。对数据缩放敏感，对参数选取敏感，大型网络需要很长的训练时间
from sklearn.neural_network import MLPClassifier
gbm0=MLPClassifier()
gbm0.fit(x_train,y_train)

y_pred = gbm0.predict(x_test)
print('Training Score: ' , gbm0.score(x_train,y_train))
print('Testing Score: ' , gbm0.score(x_test, y_test))
# print(len(y_test),len(y_pred))
print('准确率: ',accuracy_score(y_test,y_pred))#准确率，准确率是分类正确的样本占总样本个数的比例
print('精确率: ',precision_score(y_test, y_pred))#精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
print('召回率: ',recall_score(y_test, y_pred))#召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例
print("F1值: %.3f"%f1_score(y_test, y_pred))

pred_probas = gbm0.predict_proba(x_pca)[:,1] #score
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = ' MLP area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')

# K近邻，适用于小型数据集，是很好的基准模型，容易解释
from sklearn.neighbors import KNeighborsClassifier
gbm0=KNeighborsClassifier()
gbm0.fit(x_train,y_train)

y_pred = gbm0.predict(x_test)
print('Training Score: ' , gbm0.score(x_train,y_train))
print('Testing Score: ' , gbm0.score(x_test, y_test))
# print(len(y_test),len(y_pred))
print('准确率: ',accuracy_score(y_test,y_pred))#准确率，准确率是分类正确的样本占总样本个数的比例
print('精确率: ',precision_score(y_test, y_pred))#精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
print('召回率: ',recall_score(y_test, y_pred))#召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例
print("F1值: %.3f"%f1_score(y_test, y_pred))

pred_probas = gbm0.predict_proba(x_pca)[:,1] #score
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'KNeighbors area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.xlabel('FPR',fontsize=16)
plt.ylabel('TPR',fontsize=16)
plt.grid(True, linestyle='--')
plt.title('Multiple classifier ROC curve')
plt.show()