from aip import AipNlp

""" 你的 APPID AK SK """
APP_ID = ''
API_KEY = ''
SECRET_KEY = ''

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

# text='百度是一家高科技公司'
# result=client.sentimentClassify(text)
# print(result)

#-*- coding : utf-8 -*-
# coding: utf-8
# coding: unicode_escape
import pandas as pd
import codecs,sys,string,re,csv,time
import xlrd
import xlutils.copy
output=codecs.open('baiduResult.csv','w',encoding='gbk')
output1=codecs.open('baiduResult1.csv','w',encoding='gbk')
output2=codecs.open('baiduResult2.csv','w',encoding='gbk')


def append_csv(con):
    with open('baiduResult.csv', "a+", newline='') as file: # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        datas = [con]
        csv_file.writerows(datas)

rowList=[]
rowList1=[]
rowList2=[]
with open('test.csv','r',encoding='gbk',errors='ignore') as f:
    f_c=csv.reader(f)
    for row in f_c:
        if row:
            row=''.join(row)
            row.encode('utf-8').decode("utf-8")

            print(row)
            result=client.sentimentClassify(row)
            print(result['items'])

            sentiment=result['items'][0]['sentiment']
            positive_prob=result['items'][0]['positive_prob']
            negative_prob = result['items'][0]['negative_prob']
            # time.sleep()
            rowList.append(str(sentiment)+'\n')
            rowList1.append(str(positive_prob) + '\n')
            rowList2.append(str(negative_prob) + '\n')
            append_csv(str(sentiment)+'\n')

            # break
            time.sleep(0.5)
    print(rowList,rowList1,rowList2)
    output.writelines(rowList)
    output1.writelines(rowList1)
    output2.writelines(rowList2)

output.close()
output1.close()
output2.close()
