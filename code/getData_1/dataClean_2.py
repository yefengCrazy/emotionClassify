import codecs
import re
import numpy as np
import csv
output=codecs.open('output_all.csv','w')
cc=[]
cb=[]
#第一遍清洗，清除换行符等各类符号，处理完毕后，将源文件的‘力荐’等复制到output第二列
with open('comments_all.csv','r') as f:
    f_c=csv.reader(f)
    for row in f_c:
        if row:
            line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+".encode('utf-8').decode("utf8"), "",row[5])
            line = re.sub("[a-zA-Z0-9]", "", line)
        print(line)
        cc.append(line+'\n')
    print(len(cc))
    output.writelines(cc)
f.close()

