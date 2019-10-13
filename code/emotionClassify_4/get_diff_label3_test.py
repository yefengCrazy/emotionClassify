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
                    c='#FF7F0E',
                    edgecolor='#FF7F0E',#1F77B3深蓝色
                    alpha=0.5, label=label0)#Correct prediction  Partial conservative type  #1F77B4橙色
    if temp_1:
        plt.scatter(np.array(temp_1)[:, 0], np.array(temp_1)[:, 1], marker='o',
                    cmap='blues',
                    c='#0D6298',#409DDB橙色
                    edgecolor='#1F77B3', alpha=0.4, label=label1)#Preferential praise type
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
df = pd.read_csv('test_3_x_ypre1_ypre0.csv',error_bad_lines=False)

y1_pre = df.iloc[:,1]
y0_pre= df.iloc[:,2]

y1_true=np.ones(len(y1_pre))
y0_true=np.zeros(len(y0_pre))

text = df.iloc[:,3:4]
x = df.iloc[:,4:]

# print(Counter(y1_pre),Counter(y1_true))
# print(Counter(y0_pre),Counter(y0_true))
# print(text)
# print(x)

#在模型1中预测为0的数据和在模型二中预测为0的数据交叉对比

#新的想法
#1-0  0-0 原本为1的预测为0，原本为0预测为0，标准量0
#1-1  0-1 原本为1的预测为1，原本为0预测为1，标准量1
#1-0  0-1 原本为1的预测为0，原本为0预测为1，无法预测量
#1-1  0-0 偏移量，左右摇摆不定，随标记而标记
data_1_0_0_0=[]
data_0_1_1_1=[]
flag=0
sum=[]
text=[]
for index,i in enumerate(y1_pre):
    if i == 0 and y0_pre[index]==0:
        flag+=1

