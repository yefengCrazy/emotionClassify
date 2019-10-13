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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 获取数据 [1995 rows x 400 columns]
fdir = ''
df = pd.read_csv(fdir + '2000_data.csv')
y = df.iloc[:,1]
x = df.iloc[:,2:]


# # PCA降维
# ##计算全部贡献率
# n_components = 400
# pca = PCA(n_components='mle')
# pca.fit(x)
# #print pca.explained_variance_ratio_
#
# ##PCA作图
# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.axes([.2, .2, .7, .7])
# plt.plot(pca.explained_variance_, linewidth=2)
# plt.axis('tight')
# plt.xlabel('n_components')
# plt.ylabel('explained_variance_')
# plt.show()



##根据图形取100维
x_pca = PCA(n_components = 100).fit_transform(x)

print(x_pca)
print(len(x_pca))
print(len(x_pca[0]))

#划分数据集
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(
    x_pca,y,random_state=1,train_size=0.7
)

# SVM (RBF)
# using training data with 100 dimensions

# clf = svm.SVC(C = 2, probability = True)
# clf.fit(x_train,y_train)

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


pred_probas = gbm0.predict_proba(x_test)[:,1] #score
precision,recall,_=precision_recall_curve(y_test,pred_probas)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('GradientBoostingClassifier PR curve')
plt.show()

pred_probas = gbm0.predict_proba(x_pca)[:,1] #score
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.title('GradientBoostingClassifier ROC curve')
plt.legend(fontsize=16)
plt.show()

