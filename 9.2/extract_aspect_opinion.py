#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :  extract_aspect_opinion.py
@Time    :  2021/8/30 11:29
@Author  :
@Version :  1.0
@Contact :
@Desc    :  输出goods_id,comment_id,labels,labels_index,sentiment,paragraph
'''

import re
import os
import jsonlines
from utils.analysis_utils import *
from stanfordcorenlp import StanfordCoreNLP

cur_path = os.getcwd()
path  = r'C:\\Users\\11415\\Desktop\\stanford-corenlp-4.2.2\\'
nlp = StanfordCoreNLP(path)

class ProcessData():
    def __init__(self,path):
        self.path = path


    def read_aspect_vocab(self,path):
        aspect_vocab = {}
        with open(path, encoding='utf-8') as reader:
            for line in reader.readlines():
                for aspect in line.strip().split(','):
                   aspect_vocab[aspect] = len(aspect_vocab)
        return aspect_vocab

    def read_sentiment_vocab(self,path):
        sentiment_vocab = {}
        with open(path, encoding='utf-8') as reader:
            for line in reader.readlines():
                sentiment_vocab[line.strip()] = len(sentiment_vocab)
        return  sentiment_vocab

    def read_data(self):
        aspect_vocab = self.read_aspect_vocab("vocab/aspect_vocab.txt")
        deny_vocab = self.read_sentiment_vocab("vocab/deny.txt")
        positive_vocab = self.read_sentiment_vocab("vocab/positive.txt")
        negative_vocab = self.read_sentiment_vocab("vocab/negative.txt")
        os.makedirs("result/",exist_ok=True)
        writer =  jsonlines.open("result/result.jsonl", 'w')
        pattern = re.compile(r'<[^>]+>', re.S)
        punc = '！@#￥%&*（）—“：’；、。，？》《%'
        with open(self.path,encoding='utf-8') as reader:
            for line in jsonlines.Reader(reader):
                text = line['text'].lower()
                goods_id = line['goods_id']
                id = line['id']
                text = pattern.sub('', text)
                text = re.sub(r"[%s]+" % punc, "", text)
                text_tk = nlp.word_tokenize(text)
                dependency = nlp.dependency_parse(text)
                pos = nlp.pos_tag(text)
                analysis_utils = AnalysisUtils()
                ##查找aspect_opinion对
                labels,indexs = analysis_utils.get_single_sentence_lables(text_tk,aspect_vocab,dependency,pos)

                #分析情感,不在两个词典中的情感默认为-1
                sentiments = analysis_utils.analysis_sentiment(labels,positive_vocab,negative_vocab)

                #否定处理
                labels,sentiments = analysis_utils.analysis_deny(labels,indexs,dependency,text_tk,deny_vocab,sentiments)

                #得到paragrapth
                paragraphs = analysis_utils.analysis_paragraph(labels,indexs,text,text_tk)

                writer.write({
                        "goods_id":goods_id,
                        "id":id,
                        "text":text,
                        "labels":labels,
                        "sentiments":sentiments,
                        "paragraphs":paragraphs})

if __name__ =='__main__':

    process_data = ProcessData("data/14res_data_sampled.jsonl")
    process_data.read_data()
