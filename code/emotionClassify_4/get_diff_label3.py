import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plt_subplot(x_train,y_train,path,title,label0,label1):
    temp_1 = []
    temp_0 = []
    for i in range(len(x_train.tolist())):
        if y_train.tolist()[i] == 1.0:
            temp_1.append(x_train[i])
        else:
            temp_0.append(x_train[i])

    plt.figure(figsize=(8, 6))

    print(temp_0)
    print(temp_1)
    if temp_0:
        plt.scatter(np.array(temp_0)[:, 0], np.array(temp_0)[:, 1], marker='o',
                    cmap='Reds',
                    c='#0303FF',
                    edgecolor='#0303FF', alpha=1, label=label0)#Correct prediction  Partial conservative type  #1F77B4橙色
    if temp_1:
        plt.scatter(np.array(temp_1)[:, 0], np.array(temp_1)[:, 1], marker='x',
                    cmap='Reds',
                    c='#FF7F0E',
                    edgecolor='#FF7F0E', alpha=1, label=label1)#Preferential praise type
    #     #1F77B4蓝色   #FF4805红色
    plt.grid(True, linestyle='--')
    plt.title(title)
    plt.ylabel('Variable X')
    plt.xlabel('Variable Y')
    plt.legend(loc='upper right')
    plt.savefig(path)
    plt.show()
from collections import Counter
# 获取数据 [1995 rows x 400 columns]
fdir = ''
df = pd.read_csv(fdir + '3_label_54_321.csv')
df543 = pd.read_csv(fdir + '3_label_543_21.csv')

y = df.iloc[:,1]
x = df.iloc[:,2:]
y2= df543.iloc[:,1]
x2 = df543.iloc[:,2:]


yy=np.array(y)
xx=np.array(x)
yy2=np.array(y2)
xx2=np.array(x2)

print('1',Counter(yy))
print('2',Counter(yy2))

import sklearn.model_selection
#划分数据集
x_train1,x_test1,y_train1,y_test1=sklearn.model_selection.train_test_split(
    xx,yy,random_state=1,train_size=0.7
)

print('3',Counter(y_test1))

#划分数据集
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(
    xx2,yy2,random_state=1,train_size=0.7
)

print('4',Counter(y_test))

xx=x_test1.tolist()
yy=y_test1.tolist()
xx2=x_test.tolist()
yy2=y_test.tolist()


temp_x=[]
temp_y=[]

temp_x0=[]
temp_y0=[]



# plt_subplot(np.array(temp_x_x0),np.array(temp_y_y0),'543_21.png','classification 3','ZERO','ONE')


