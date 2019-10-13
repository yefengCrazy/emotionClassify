import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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

import time
# 获取数据 [1995 rows x 400 columns]
fdir = ''
df = pd.read_csv(fdir + '2000_data_text.csv',encoding='utf-8')
y = df.iloc[:,1]
x = df.iloc[:,3:]
x_text=df.iloc[:,2]

##根据图形取100维
x_pca = PCA(n_components = 100).fit_transform(x)

#划分数据集
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(
    x_pca,y,random_state=1,train_size=0.7
)

#开始划分数据

#先从训练集开始，分成两个模型(543,21)(54,321)
y_train0=[] #(54,321)
y_train1=[] #(543,21)
for i in y_train:
    if i == 3:
        y_train0.append(0)
        y_train1.append(1)
    else:
        y_train0.append(i)
        y_train1.append(i)

#测试集数据划分
y_test3_true0=[]
y_test3_true1=[]
x_test3=[]
x_test3_text_index=[]
x_test3_text=[]
length3=0  #标记测试集样本中有多少个3样本，便于全0化1化
# print(type(y_test))
for index1,i in enumerate(y_test):
    if i == 3:
        # print(y_test.index[index1])#在测试集样本中，标签为3的样本，找到它的index
        x_test3_text_index.append(y_test.index[index1])
        x_test3.append(x_test[index1])
print(len(x_test3_text_index))
#通过索引查找对于文本
#对总样本集20655遍历，如果一条在3样本集中，就记录下文本内容
for index1,i in enumerate(x_text):
    if x_text.index[index1] in x_test3_text_index:
        x_test3_text.append(i)

print(len(x_test3_text))

length3=len(x_test3)#测试集中3样本对应的x标签
# print(length3)
y_test3_true1=np.ones(length3)#测试集中3样本(543，21)对应的y标签
y_test3_true0=np.zeros(length3)#测试集中3样本(54，321)对应的y标签

# sys.exit(0)

# print('采样前',Counter(y_train1))
# from imblearn.over_sampling import RandomOverSampler as  ros
# #https://blog.csdn.net/weixin_44871660/article/details/90600522
# from imblearn.combine import SMOTEENN
# ros = SMOTEENN(random_state=0)
# X_resample,y_resample = ros.fit_resample(x_train,y_train1)
# # sys.exit(0)
# x_train1=X_resample
# y_train1=y_resample
# print('采样后',Counter(y_train))

#训练数据,针对3样本，测试性能
# 神经网络，可以构建非常复杂的模型，特别是对于大型数据而言。对数据缩放敏感，对参数选取敏感，大型网络需要很长的训练时间
from sklearn.neural_network import MLPClassifier
gbm0=MLPClassifier()
gbm0.fit(x_train,y_train1)#(543，21)模型
y_pred1 = gbm0.predict(x_test3)
print('543,21模型')

# print('采样前',Counter(y_train1))
# from imblearn.over_sampling import RandomOverSampler as  ros
# #https://blog.csdn.net/weixin_44871660/article/details/90600522
# from imblearn.combine import SMOTEENN
# ros = SMOTEENN(random_state=0)
# X_resample,y_resample = ros.fit_resample(x_train,y_train0)
# # sys.exit(0)
# x_train0=X_resample
# y_train0=y_resample
# print('采样后',Counter(y_train0))

# 神经网络，可以构建非常复杂的模型，特别是对于大型数据而言。对数据缩放敏感，对参数选取敏感，大型网络需要很长的训练时间
gbm0.fit(x_train,y_train0)
y_pred0 = gbm0.predict(x_test3)
print('54,321模型')



# write in file导出模板
# 导出一个文件，导出值为测试集中3样本的x值，y在两个模型下的预测标签，第一列为(543，21)模型的预测标签，第二列为(54，321)预测标签，第三列为文本，第四列为x值
df_x = pd.DataFrame(x_test3)
df_y1 = pd.DataFrame(y_pred1)
df_y0 = pd.DataFrame(y_pred0)
df_text=pd.DataFrame(x_test3_text)
data = pd.concat([df_y1, df_y0, df_text, df_x], axis=1)
# # print data
data.to_csv('test_3_x_ypre1_ypre0.csv',encoding="utf_8_sig")
# # data.to_csv(fdir + '2000_data_3.csv')


#验证在分完训练集和测试集后，将3样本化为两个模型来求较差的可行性
# # 神经网络，可以构建非常复杂的模型，特别是对于大型数据而言。对数据缩放敏感，对参数选取敏感，大型网络需要很长的训练时间
# from sklearn.neural_network import MLPClassifier
# gbm0=MLPClassifier()
# gbm0.fit(x_train,y_train)#(543，21)模型
#
# y_test_temp=[]
# for index,i in enumerate(y_test):
#     if i == 3:
#         y_test_temp.append(1)
#     else:
#         y_test_temp.append(i)
#
# y_pred1 = gbm0.predict(x_test)
# print('543,21模型')
# print(Counter(y_test_temp),Counter(y_pred1))
# print('准确率: ',accuracy_score(y_test_temp,y_pred1))#准确率，准确率是分类正确的样本占总样本个数的比例
# print('精确率: ',precision_score(y_test_temp, y_pred1))#精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
# print('召回率: ',recall_score(y_test_temp, y_pred1))#召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例
# print("F1值: %.3f"%f1_score(y_test_temp, y_pred1))