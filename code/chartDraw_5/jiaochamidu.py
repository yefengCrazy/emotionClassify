import numpy as np
import matplotlib.pyplot as plt
import random
N = 50
# menMeans = (2, 0, 2, 0, 1,1,3)
# womenMeans = (25, 32, 34, 20, 25,1,2)
random.seed(0)
r1=[]
r2=[]
for i in range(50):
    r1.append(random.randint(0,2))
    r2.append(random.randint(0,2))
print(r1)
print(r2)
menMeans=r1
womenMeans=r2
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width)
p2 = plt.bar(ind, womenMeans, width,
             # bottom=menMeans
             )

plt.ylabel('Scores')
plt.title('Scores by group and gender')
list1=[]
for i in range(50):
    list1.append(i+1)
print(list1)
# plt.xticks(ind, (list1))
plt.yticks(np.arange(0, 3, 1))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()