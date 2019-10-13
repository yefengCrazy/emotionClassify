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

##根据图形取100维
# x_pca = PCA(n_components = 100).fit_transform(x)
pca=PCA(n_components=100,random_state=0)
x_pca=pca.fit_transform(x)


#划分数据集
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(
    x_pca,y,random_state=1,train_size=0.7
)
#获取测试集数据
df=pd.read_csv(fdir + '2000_data_3.csv')
xx=df.iloc[:,2:]
x_test=pca.fit_transform(xx)

# SVM (RBF)
# using training data with 100 dimensions

# clf = svm.SVC(C = 2, probability = True)
# clf.fit(x_train,y_train)
# 神经网络，可以构建非常复杂的模型，特别是对于大型数据而言。对数据缩放敏感，对参数选取敏感，大型网络需要很长的训练时间
from sklearn.neural_network import MLPClassifier
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
ros = SMOTETomek(random_state=0)
x_train,y_train = ros.fit_resample(x_train,y_train)

gbm0=MLPClassifier()
gbm0.fit(x_train,y_train)

y_pred = gbm0.predict(x_test)
print(Counter(y_pred))
print(y_pred)
print('Training Score: ' , gbm0.score(x_train,y_train))

pca2=PCA(n_components=2,random_state=0)
x_2=pca2.fit_transform(xx)

# write in file
df_x = pd.DataFrame(x_2)
df_y = pd.DataFrame(y_pred)
data = pd.concat([df_y,df_x],axis = 1)
#print data
# data.to_csv(fdir + '2000_data.csv')
data.to_csv(fdir + '3_label_543_21.csv')


sys.exit(0)

print('Testing Score: ' , gbm0.score(x_test, y_test))
# print(len(y_test),len(y_pred))
print('准确率: ',accuracy_score(y_test,y_pred))#准确率，准确率是分类正确的样本占总样本个数的比例
print('精确率: ',precision_score(y_test, y_pred))#精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
print('召回率: ',recall_score(y_test, y_pred))#召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例
print("F1值: %.3f"%f1_score(y_test, y_pred))

plt.figure(figsize=(10,6))
plt.legend(loc="lower right",fontsize=16)
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
plt.title('MLPClassifier PR curve')
plt.show()

plt.figure(figsize=(10,6))
plt.legend(loc="lower right",fontsize=16)
pred_probas = gbm0.predict_proba(x_pca)[:,1] #score
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.title('MLPClassifier ROC curve')
plt.show()

