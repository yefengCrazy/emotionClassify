import matplotlib as mpl
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 生成数据
labels = ['正向情感', '负向情感', '中性情感']
share = [619, 267,67]
plt.xticks(fontsize=30)

plt.yticks(fontsize=30)
# 设置分裂属性
explode = [0.07, 0.07, 0.07]

# 分裂饼图
plt.pie(share, explode = explode,
        labels = labels, autopct = '%3.1f%%',
        startangle = 45, shadow = True,
        colors = ['c', 'r', 'gray'])

# 标题
plt.title('样本三数据占比')

plt.show()