import hashlib
import time
import random
import string
import urllib
import sys

import requests
# import md5sign
from bs4 import BeautifulSoup
import json
import sys
import importlib,sys
importlib.reload(sys)

# sys.setdefaultencoding('utf-8')

def get_params(plus_item):
    '''请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效）'''
    t = time.time()
    time_stamp=int(t)

    '''请求随机字符串，用于保证签名不可预测'''
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 10))

    '''应用标志，这里修改成自己的id和key'''
    app_id=''
    app_key=''

    '''值使用URL编码，URL编码算法用大写字母'''
    text1=plus_item
    text=urllib.request.quote(text1.encode().decode(sys.stdin.encoding).encode('utf8')).upper()

    # print(text)
    '''拼接应用密钥，得到字符串S'''
    sign_before='app_id='+app_id+'&nonce_str='+nonce_str+'&text='+text+'&time_stamp='+str(time_stamp)+'&app_key='+app_key

    '''计算MD5摘要，得到签名字符串'''
    m=hashlib.md5()
    m.update(sign_before.encode('utf-8'))
    sign=m.hexdigest()
    sign=sign.upper()

    params='app_id='+app_id+'&time_stamp='+str(time_stamp)+'&nonce_str='+nonce_str+'&sign='+sign+'&text='+text

    return params

def get_content(plus_item):
    url = "https://api.ai.qq.com/fcgi-bin/nlp/nlp_textpolar"  # API地址
    params = get_params(plus_item)#获取请求参数
    url=url+'?'+params#请求地址拼接
    # print(url)
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'lxml')
        allcontents=soup.select('body')[0].text.strip()
        allcontents_json=json.loads(allcontents)#str转成dict
        print(allcontents_json)
        return allcontents_json["data"]["polar"],allcontents_json["data"]["confd"],allcontents_json["data"]["text"],allcontents_json['msg']

    except Exception as e:
        print('a', str(e))
        return (0,0,0)


def append_csv(polar,confd):
    with open('tengxunResult.csv', "a+", newline='') as file: # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        datas = [str(polar)]
        datas1 = [str(confd)]
        csv_file.writerows(zip(datas,datas1))

import pandas as pd
import codecs,sys,string,re,csv,time
import xlrd
import xlutils.copy
output=codecs.open('baiduResult.csv','w',encoding='gbk')
output1=codecs.open('baiduResult1.csv','w',encoding='gbk')
output2=codecs.open('baiduResult2.csv','w',encoding='gbk')

def getCon():
    a=[]
    with open('test.csv', 'r', encoding='gbk', errors='ignore') as f:
        f_c = csv.reader(f)
        for row in f_c:
            if row:
                row = ''.join(row)
                row.encode('utf-8').decode("utf-8")
                a.append(row)
                # print(row)
    return a
import time
if __name__ == '__main__':
    # polar,confd,text=get_content('今天天气真好')

    aList=[]
    confdList=[]
    aList=getCon()
    for i in aList:

        i=i[:60]
        print(i)
        ok=''
        while ok!='ok':
            polar, confd, text,ok = get_content(i)
            time.sleep(2)
            if ok == 'sign invalid':
                polar='error'
                confd='error'
                break;

        append_csv(str(polar),str(confd))
        time.sleep(2)
        # print('情感倾向：'+str(polar)+'\n'+'程度：'+str(confd)+'\n'+'文本：'+text)