import codecs
import re
import numpy as np
import csv
output=open('output2.csv','w',newline='')
out1=open('1_out.txt','w')
out0=open('0_out.txt','w')
out3=open('3_out.txt','w')
# out2=open('2_out.txt','w')
f_o=csv.writer(output)
cc=[]
cb=[]
#第二遍清洗,将对应0，1合并到csv文件中,拆分成两个txt文件(1，0)，便于后面分析
with open('output_1_all.csv','r') as f:
    f_c=csv.reader(f)
    # for row in f_c:
    #     #     if row:
    #     #         line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+".encode('utf-8').decode("utf8"), "",row[0])
    #     #     print(line)
    #     #     cc.append(line+'\n')
    #     # print(len(cc))
    #     # output.writelines(cc)
    print(type(f_c))
    for row in f_c:
        # print(row[0],row[1])
        cc=[]
        if row[0]:
            cc.append(row[0])
            if row[1] == '力荐'or row[1]=='推荐':
                cc.append('1')
                f_o.writerow(cc)
                out1.writelines(row[0]+'\n')
            elif ( row[1]=='较差' or row[1]=='很差'):
                cc.append('0')
                f_o.writerow(cc)
                out0.writelines(row[0]+'\n')
            elif ( row[1]=='还行'):
                cc.append('3')
                f_o.writerow(cc)
                out3.writelines(row[0]+'\n')
        # f_o.writerow(cc)
        cb.append(cc)
    # for row in f_c:
    #     # print(row[0],row[1])
    #     cc=[]
    #     if row[0]:
    #         cc.append(row[0])
    #         if row[1] == '力荐':
    #             cc.append('0')
    #             f_o.writerow(cc)
    #             out0.writelines(row[0]+'\n')
    #         elif (row[1]=='推荐'):
    #             cc.append('1')
    #             f_o.writerow(cc)
    #             out1.writelines(row[0]+'\n')
    #         elif row[1]=='还行':
    #             cc.append('2')
    #             f_o.writerow(cc)
    #             out2.writelines(row[0]+'\n')
    #
    #     # f_o.writerow(cc)
    #     cb.append(cc)
    print(cb)
f.close()
output.close()

out0.close()
out1.close()
out3.close()
