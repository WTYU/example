#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:59:58 2019

@author: wutiyu
"""

import os
import pickle
import jieba #中文結巴函式庫
import json
import operator
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from datetime import datetime
from collections import Counter

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

#微軟正黑體
font_path = 'msjh.ttc'
#設定字體相關屬性（字型 粗體 字體大小）
font = font_manager.FontProperties(fname='msjh.ttc',
                                   weight='bold',
                                   style='normal', size=16)

#設定主辭典路徑（路徑中的 .. 即可回到上一層之資料夾）
jieba.set_dictionary('../jieba_data/dict.txt.big')
jieba.load_userdict('../jieba_data/userdict.txt')
stopwords = []
#讀取stopwords.txt檔案 並放入stopwords陣列
#原先讀檔時會有 encoding='UTF-8'如在python2的環境下則不需此參數 
with open('../jieba_data/stopwords.txt', 'r') as file:
    for each in file.readlines():
        stopwords.append(each.strip())
    stopwords.append(' ')

stopwords[:10] #印出前十筆陣列值

#用load方式載入提取new_talk.pkl
with open('../crawler/data/new_talk.pkl', 'rb') as f:
    data = pickle.load(f)

#將data此list倒過來
data = data[::-1]
contents = [news['content'] for news in data]

#開啟並讀取name.txt檔 並以換行符號為切割點 將切割結果存入name
names = []
with open('../data/names.txt', 'r') as f:
    names = f.read().split('\n')
    
##開啟並讀取events.txt檔 並以換行符號為切割點 將切割結果存入events
events = []
with open('../data/events.txt', 'r') as f:
    events = f.read().split('\n')

'''此函式主要在做移除傳入字串中的英文及各式符號
   並將找到此些英文或是符號的地方替換成空格 最後
   再傳回移除後的字串結果
'''
def remove_punctuation(content_string, user_pc=False):
    if(user_pc):
        punctuation = user_pc
    else:
        punctuation=list("!@#$%^&*()_+=-[]`~'\"|/\\abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.;{}\r\xa0\u3000、，。「」！？；：<>")
        
    for p in punctuation:
        content_string = content_string.replace(p, " ")
    return(content_string)

'''將傳入的 word_dict 中與 stopwords 一樣的元素移除
   最後再將移除完的word_dict回傳
'''
def remove_stopwords_from_dict(word_dict, stopwords):
    for w in stopwords:
        word_dict.pop(w, word_dict)
    return word_dict

'''計算lcut中各元素所對應的個數 最後再丟給remove_stopwords_from_dict
   去移除其中與stopwords一樣的元素
'''
def lcut_to_dict(lcut):
    word_dict = dict(Counter(lcut))
#     word_dict.pop(' ')
    return(remove_stopwords_from_dict(word_dict, stopwords))

'''以字典中各個key所對應的值做排序 如果單單只用sorted來排
   在python中字典還會是亂的 無法真正排好 因此需要後面的參數設定
'''
def sort_dict_by_values(d):
    return(sorted(d.items(), key=lambda x: x[1], reverse=True))

'''回傳值為一個generator lambda式子可以寫成 (news for news in news_list if keyword in news)
   意即 當news_list中有包含keyword就放入其中
'''
def news_containing_keyword(keyword, news_list):
    return list(filter(lambda news: keyword in news, news_list))

'''與上方函式相同 差別只在於判斷的範圍改成data
'''
def data_containing_keyword(keyword, data):
    return list(filter(lambda news: keyword in news['cutted_dict'].keys(), data))

'''判斷news_list中是否有包含keywords中的元素
'''
def news_containing_keywords(keywords, news_list):
    news = news_list
    for keyword in keywords:
        news = news_containing_keyword(keyword, news)
        
    return news

# add cutted dict to each news
for i in range(len(data)):
    current_content = data[i]['content']
    current_cutted = jieba.lcut(remove_punctuation(current_content))
    data[i]['cutted_dict'] = lcut_to_dict(current_cutted)

'''結算出content中各字詞依照出現頻率排序的結果'''
def get_coshow(content):
    coshow_dict = {}
    cat_content = ' '.join(contents[:100])#放入前一百個
    clean_content = remove_punctuation(cat_content)#剔除英文及各式符號
    cut_content = jieba.lcut(clean_content)#clean_content經過斷詞之後的list
    cut_content = list(filter(lambda x: x!=' ', cut_content))#去除剛剛斷詞完中list的空白
    for i in range(len(cut_content)-1):
        wcw = cut_content[i] + cut_content[i+1]
    #     print(wcw)
        try:
            coshow_dict[wcw] = coshow_dict[wcw] + 1
        except:
            coshow_dict[wcw] = 1

    sdbv = sort_dict_by_values(coshow_dict)#排序字典
    return sdbv#回傳排序好的結果

'''去除空白並做斷詞 回傳去除斷字符後的list'''
def get_cutted_dict(list_of_news):
    cat = ' '.join(list_of_news)
    cat = remove_punctuation(cat)#剔除英文及各式符號
    cutted = jieba.lcut(cat)
    return lcut_to_dict(cutted)

'''找出字串長度大於二小於一千中前n個字元'''
def first_n_words(cutted_dict, n, word_len=2, to=1000):
    sdbv = sort_dict_by_values(cutted_dict)
    return list(filter(lambda x: len(x[0])>=word_len and len(x[0])<=to, sdbv))[:n]

