import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

s=np.array(['SVM','Bays','LogisticRegression','DecisionTree','GradientBoosting','RandomForest','MLP','KNeighbors'])
# y1=np.array([0.941,0.503,0.940,0.885,0.941,0.940,0.930,0.939])
# y2=np.array([0.798,0.498,0.789,0.688,0.796,0.766,0.754,0.773])

# y1=np.array([0.941,0.955,0.942,0.944,0.942,0.944,0.948,0.942])
# y2=np.array([0.798,0.835,0.800,0.814,0.798,0.812,0.838,0.817])

# y1=np.array([0.999,0.596,0.998,0.933,0.999,0.996,0.979,0.996])
# y2=np.array([0.997,0.454,0.977,0.784,0.993,0.916,0.853,0.919])

# y1=np.array([0.970,0.653,0.969,0.938,0.970,0.969,0.963,0.968])
# y2=np.array([0.886,0.589,0.880,0.799,0.885,0.861,0.846,0.865])

#SMOTE
# y1=np.array([0.893	,0.607	,0.827	,0.885	,0.879	,0.923	,0.950	,0.749])
# y2=np.array([0.890	,0.631	,0.813	,0.896	,0.878	,0.931	,0.953	,0.733])
# y3=np.array([0.789	,0.549	,0.752	,0.758	,0.781	,0.785	,0.837	,0.599])
#SMOTETomek
# y1=np.array([0.893	,0.609	,0.825	,0.881	,0.879	,0.923	,0.955	,0.755])
# y2=np.array([0.890	,0.632	,0.814	,0.888	,0.878	,0.932	,0.950	,0.736])
# y3=np.array([0.788	,0.544	,0.753	,0.759	,0.782	,0.778	,0.835	,0.595])
#SMOTEENN
y1=np.array([0.803	,0.559	,0.680	,0.806	,0.722	,0.807	,0.888	,0.570])
y2=np.array([0.786	,0.588	,0.649	,0.811	,0.724	,0.827	,0.891	,0.548])
y3=np.array([0.547	,0.493	,0.425	,0.599	,0.479	,0.509	,0.676	,0.301])

plt.figure(figsize=(15,8))
bar_width=0.25
# Y轴数据使用range(len(x_data), 就是0、1、2...
plt.barh(y=range(len(s)), width=y1, label=u'模型一',
         color='#E5624A', alpha=0.8, height=bar_width)
# Y轴数据使用np.arange(len(x_data))+bar_width,
# 就是bar_width、1+bar_width、2+bar_width...这样就和第一个柱状图并列了
plt.barh(y=np.arange(len(s)) + bar_width, width=y2,
         label=u'模型二', color='#4DD3E5', alpha=0.8, height=bar_width)
plt.barh(y=np.arange(len(s)) + bar_width*2, width=y3,
         label=u'模型三', color='#7E860D', alpha=0.8, height=bar_width)

# # 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# for y, x in enumerate(y1):
#     plt.text(x + 5000, y - bar_width / 3, '%s' % x, ha='center', va='bottom')
# for y, x in enumerate(y2):
#     plt.text(x + 5000, y + bar_width / 3, '%s' % x, ha='center', va='bottom')
# for y, x in enumerate(y3):
#     plt.text(x + 5000, y + bar_width / 3, '%s' % x, ha='center', va='bottom')
# 为Y轴设置刻度值
plt.yticks(np.arange(len(s)) + bar_width / 3 + 0.15, s)

# 设置标题
#精确度，查准率，召回率，F1值(F1-Measure)
plt.title(u"F1值(F1-Measure)")
# 为两条坐标轴设置名称
plt.xlabel(u"占比",)
# plt.ylabel(u"机器学习方式")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 显示图例
plt.legend(loc=0, numpoints=1)

leg = plt.gca().get_legend()

ltext = leg.get_texts()

plt.setp(ltext, fontsize='small')
plt.show()