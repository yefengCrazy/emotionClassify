import numpy as np
import matplotlib.pyplot as plt
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
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import average_precision_score
# import some data to play with
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
from imblearn.over_sampling import RandomOverSampler as  ros
#https://blog.csdn.net/weixin_44871660/article/details/90600522
# from collections import Counter
# from imblearn.combine import SMOTEENN
# ros = SMOTEENN(random_state=0)
# X_resample,y_resample = ros.fit_resample(x_train,y_train)
# print('2',Counter(y_resample))
# # sys.exit(0)
# x_train=X_resample
# y_train=y_resample
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear',probability=True)
y_pred = classifier.fit(x_train, y_train).predict(x_test)

#pr曲线
from sklearn.metrics import average_precision_score
# pred_probas = clf.predict_proba(x_test)[:,1] #score
pred_probas = classifier.predict_proba(x_test)[:,1] #score
precision,recall,_=precision_recall_curve(y_test,pred_probas)
auc_svm=average_precision_score(y_test,pred_probas)
# plt.step(recall,precision, color='b', alpha=0.2,
#          where='post',label='321')
# plt.plot(recall,precision,color='r',lw=1)
plt.plot(recall,precision,color='r',label='SVM (AUC: %.3f)'% auc_svm,lw=1)
# plt.fill_between(recall, precision, step='post', alpha=0.2,
#                  color='b')


gnb=GNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)

pred_probas = gnb.predict_proba(x_test)[:,1] #score
precision,recall,_=precision_recall_curve(y_test,pred_probas)
auc_gnb=average_precision_score(y_test,pred_probas)
# plt.step(recall, precision,  color='r', alpha=0.2,
#          where='post',label='123')
plt.plot(recall,precision,color='b',label='GNB (AUC: %.3f)' % auc_gnb,lw=1)
# plt.plot(fpr, tpr, label = 'KNeighbors area = %.2f' % roc_auc)
# plt.plot(recall_svm,precision_svm,,linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(fontsize=12)
plt.show()