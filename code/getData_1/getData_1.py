
import re
import threading
import json
import requests
from selenium import webdriver
from requests.exceptions import RequestException
import random
from urllib import request
import time
from bs4 import BeautifulSoup
import codecs
import urllib.request
from urllib.request import urlopen
import pandas as pd
import re
import string
headers=[
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
    "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
    'Opera/9.25 (Windows NT 5.1; U; en)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
    'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
    "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
    "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 "
]
ip=[]

#soup，find_all返回的是RS，使用[]来访问其中的tag，而tag使用['']来访问，tag可直接跟a，p，span等标签

def getbs4(url):
    try:
        time.sleep(random.randint(3,7))
        header={'User-Agent':random.choice(headers)}
        print(header)
        # proxy_support=request.ProxyHandler(pr)
        req=urllib.request.Request(url,headers=header)
        response=urlopen(req)
        soup=BeautifulSoup(response,'lxml')
        # print(bsobj)
        response.close()
        return soup
    except Exception as e:
        print(url,':解析错误...')
        print(e)
    return;


def get_content(url,data):#读取指定内容块里面的信息

    list = []
    soup = BeautifulSoup(data, 'lxml')
    vote = soup.find(class_='vote-count').get_text()
    content = soup.find(class_='short').get_text()
    name = soup.find('a')['title']

    date_list = soup.find('span',{'class':'comment-info'})
    date = ''
    date_list=str(date_list)
    date_list=date_list.split('\n')

    if(len(date_list)==5):
        date=(date_list[3][6:-7])
    else:
        date=(date_list[2][6:-7])



    data_cid = soup.find('li')['data-cid']
    try:
        rating=soup.find_all('span')[3]['title']
    except:
        rating='null'
    finally:
        list = [data_cid, name, rating, date, vote, content]
    return list;

def csv_file_eachpage(path,con):#1成功，0失败
    flag=0
    try:
        data = pd.core.frame.DataFrame(con)
        data.to_csv(path,mode='a',header=False,index=None,encoding='gbk')
        flag=1
    except:
        flag=0
    return flag;

def book_one(url):#读取一本书的评论

    url1=url+'comments/hot?p='
    wrong_data=0
    valid_data=0
    for page in range(1,6):
        url=url1+str(page)

        lists_info=[]
        lists_con=[]
        soup=getbs4(url)

        # url='demo.html'
        # with open(url,'r',encoding='utf-8') as f:
        #     l=f.read()
        # soup=BeautifulSoup(l,'lxml')

        lists_info = soup.find_all(class_='comment-item')
        # print(lists_info)
        index=1
        for list in lists_info:#一页数据共20条评论，对每一条评论插入lists_con里面,类似于'™'这样的字符csv，gbk会出错，所以单条插入

            #lists_con.append(get_content(url,str(list)))
            info=get_content(url,str(list))
            flag=csv_file_eachpage('comments5.csv',[info])
            if(flag==1):
                valid_data+=1
                print('插入第 ',valid_data,' 数据...')
            else:
                wrong_data+=1
                str1=str(page)+' '+str(index)+'\n'
                with open('wrong_log.txt','a+',encoding='utf-8') as f:
                    f.write(str1)
                print('第 ',page,' 页 ',index,' 条评论错误')
                f.close()
            index+=1
        print('第 ',page,' 页内容读取完毕...')



    print('well done...')
    print('wrongdata:',wrong_data)

def get_tag():

    '''

    福尔摩斯探案全集（上中下）
    [英] 阿·柯南道尔 / 丁钟华 等 / 群众出版社 / 1981-8 / 53.00元/68.00元
    :return:
    '''
    taglists=[]
    for index in range(200,226,25):
        url='https://book.douban.com/top250?start='+str(index)
        # url='demo1.html'
        # with open(url,'r',encoding='utf-8') as f:
        #     l=f.read()
        # soup=BeautifulSoup(l,'lxml')
        soup=getbs4(url)

        top_all=soup.find_all('td',{'valign':'top'})

        for index0,top in enumerate(top_all):
            if index0 % 2==0:
                url=top.a['href']
                try:
                    book_one(url)
                except Exception as e:
                    print(url,' ',e)



        # for index1,top in enumerate(top_all):
        #     name,info,rating,pingjia,quote='','','','',''
        #     list_temp=[]
        #     if index1%2:
        #         try:
        #             # print(index,top)
        #             # print(type(str(top)))
        #             soup1=BeautifulSoup(str(top),'lxml')
        #
        #             name=soup1.find('div',{'class':'pl2'}).a['title']
        #             info=soup1.find('p',{'class':'pl'}).string
        #             rating=soup1.find('span',{'class':'rating_nums'}).string
        #             pingjia=re.findall('\d+',soup1.find('span',{'class':'pl'}).string)[0]
        #             quote=soup1.find('span',{'class':'inq'}).string
        #         except Exception as e:
        #             print(index+1,' 页 ',int(index1%2)+1,' 条数据错误')
        #             print(e)
        #         finally:
        #             list_temp.append(name)
        #             list_temp.append(info)
        #             list_temp.append(rating)
        #             list_temp.append(pingjia)
        #             list_temp.append(quote)
        #             print(list_temp)
        #             flag=csv_file_eachpage('data.csv',[list_temp])
        #             if(flag==1):
        #                 print(index,' 页',int(index1%2)+1,'条插入成功...')













    return;

def main():

    tag='1770782'
    url1=''

    get_tag()
    # book_one(tag)

    return;


if __name__ == '__main__':
    main()