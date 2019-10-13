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

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
# 获取数据 [1995 rows x 400 columns]
fdir = ''
df = pd.read_csv(fdir + '2000_data.csv')
y = df.iloc[:,1]
x = df.iloc[:,2:]

##根据图形取100维
# x_pca = PCA(n_components = 100).fit_transform(x)
pca = PCA(n_components= 250,whiten= True,svd_solver='randomized')
pca.fit(x)
gxl = pca.explained_variance_ratio_
print(sum(gxl))
print(pca)
exit()
#划分数据集
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(
    x_pca,y,random_state=1,train_size=0.7
)

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

print(Counter(y_train.tolist()))
print(Counter(y_test.tolist()))

# #应用SMOTE + ENN  训练集
# sm=SMOTEENN(random_state=42)
# X_resampled, y_resampled_SMOTEENN=sm.fit_sample(x_train, y_train.tolist())
# print('2',Counter(y_resampled_SMOTEENN))

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
#pr曲线

pred_probas = clf.predict_proba(x_test)[:,1] #score
precision,recall,_=precision_recall_curve(y_test,pred_probas)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('SVM PR你 curve')
plt.show()

pred_probas = clf.predict_proba(x_pca)[:,1] #score
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.title('SVM ROC你 curve')
plt.show()

