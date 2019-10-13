#!/usr/bin/env python
# -*- coding: utf-8  -*-
#从词向量模型中提取文本特征向量
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import logging
import os.path
import codecs,sys
import numpy as np
import pandas as pd
import gensim

# 返回特征词向量
def getWordVecs(wordList,model):
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        #print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')
    

# 构建文档词向量 
def buildVecs(filename,model):
    fileVecs = [[],[]]
    with codecs.open(filename, 'rb', encoding='utf-8') as contents:
        for line in contents:
            logger.info("Start line: " + line)
            wordList = line.split(' ')
            vecs = getWordVecs(wordList,model)
            #print vecs
            #sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                vecsArray = sum(np.array(vecs))/len(vecs) # mean
                #print vecsArray
                #sys.exit()
                fileVecs[0].append(vecsArray)
                line=line.encode('utf-8').decode('utf-8')
                fileVecs[1].append(line)
    return fileVecs   

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    # load word2vec model
    fdir = '/Users/yefeng/Desktop/lunwen/douban/senti_analysis_master/'

    inp = fdir + '1.text.vector'
    model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
    
    posInput = buildVecs(fdir + '2000_pos_cut_stopword.txt',model)
    negInput = buildVecs(fdir + '2000_neg_cut_stopword.txt',model)
    Input3 = buildVecs(fdir + '2000_temp_cut_stopword.txt', model)

    print()

    #将'还行'增加标签为'3'的数据
    temp_list=[]
    for i in range(len(Input3[0])):#构建3的分类标签
        temp_list.append(int('3'))
    temp_np=np.array(temp_list)
    # use 1 for positive sentiment， 0 for negative
    Y = np.concatenate((np.ones(len(posInput[0])), np.zeros(len(negInput[0])),temp_np))
    # Y = np.concatenate((np.ones(len(Input2))))
    # Y=(np.ones(len(Input2)))
    print(len(Y))

    X = posInput[0][:]
    for neg in negInput[0]:
        X.append(neg)
    for temp in Input3[0]:
        X.append(temp)
    X = np.array(X)

    #保存对应文本
    text=posInput[1][:]
    for neg in negInput[1]:
        text.append(neg)
    for temp in Input3[1]:
        text.append(temp)
    text = np.array(text)

    print(text)

    # write in file   
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    df_text = pd.DataFrame(text)
    data = pd.concat([df_y,df_text,df_x],axis = 1)
    #print data
    data.to_csv(fdir + '2000_data_text.csv',encoding="utf_8_sig")
    # data.to_csv(fdir + '2000_data_3.csv')

    


