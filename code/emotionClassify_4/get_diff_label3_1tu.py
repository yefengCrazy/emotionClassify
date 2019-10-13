import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sns
# sns.set()



def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像

    plt.title(title)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plt_subplot(x_train,y_train,path,title):
    temp_1 = []
    temp_0 = []
    for i in range(len(x_train.tolist())):
        if y_train.tolist()[i] == 1.0:
            temp_1.append(x_train[i])
        else:
            temp_0.append(x_train[i])

    plt.figure(figsize=(8, 6))

    if temp_0:
        plt.scatter(np.array(temp_0)[:, 0], np.array(temp_0)[:, 1], marker='o',
                    cmap='Reds',
                    c='#0303FF',
                    edgecolor='#0303FF', alpha=0.5, label='Error prediction')#Correct prediction  Partial conservative type  #1F77B4橙色
    if temp_1:
        plt.scatter(np.array(temp_1)[:, 0], np.array(temp_1)[:, 1], marker='o',
                    cmap='Reds',
                    c='#FF7F0E',
                    edgecolor='#FF7F0E', alpha=0.5, label='Correct prediction')#Preferential praise type
    #     #1F77B4蓝色   #FF4805红色
    # plt.grid(True, linestyle='--')
    plt.title(title)
    plt.ylabel('Variable X')
    plt.xlabel('Variable Y')
    plt.legend(loc='upper right')
    plt.savefig(path)
    plt.show()

# 获取数据 [1995 rows x 400 columns]
fdir = ''
df = pd.read_csv(fdir + '3_label_54_321.csv')
df543 = pd.read_csv(fdir + '3_label_543_21.csv')

y = df.iloc[:,1]#54分类的预测标签
x = df.iloc[:,2:]
y2= df543.iloc[:,1]#543分类的预测标签
x2 = df543.iloc[:,2:]

# #预测标签合并在一起
# yy54_pre=np.array(y,dtype='int')
# yy543_pre=np.array(y2,dtype='int')
# yyall_pre=np.concatenate((np.array(y,dtype='int'),np.array(y2,dtype='int')))
# print(yyall_pre,len(yyall_pre))
#
# #真实值的标签合并在一起
# value0=np.ones(len(yy54_pre),dtype='int')
# value1=np.zeros(len(yy543_pre))
# value=np.concatenate((np.ones(len(yy54_pre),dtype='int'),np.zeros(len(yy543_pre),dtype='int')))
# print(value,len(value))
from collections import Counter
#单个模型
yyall_pre=np.array(y,dtype='int')
value=np.zeros(len(yyall_pre))
classnames=['Conservative','Praise' ]
classnames=np.array(classnames)
C2=confusion_matrix(value.tolist(),yyall_pre.tolist())
plot_confusion_matrix(C2,classnames,'Confusion Matrix(54-321)')
plt.show()
print(Counter(yyall_pre))
# plt_subplot(np.array(xx_temp),np.array(yy_temp),'543_21.png','classification 3')
#单个模型
yyall_pre=np.array(y2,dtype='int')
value=np.ones(len(yyall_pre))
classnames=['Conservative','Praise' ]
classnames=np.array(classnames)
C2=confusion_matrix(value.tolist(),yyall_pre.tolist())
plot_confusion_matrix(C2,classnames,'Confusion Matrix(543-21)')
plt.show()
print(Counter(yyall_pre))


