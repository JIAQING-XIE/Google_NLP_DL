import re
import numpy as np
import jsonlines
import args
from utils import print_dict, dict2file, read_dict_from_file
from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def get_text_field(json_file):
    # task 1: extract text field from each json record
    tmp = np.array([])
    with jsonlines.open(json_file) as reader:
        for object in reader:
            tmp = np.append(tmp, object['text'])
    return tmp

def preprocess(text_array):
    # task 2: strip empty space at leftmost and rightmost & html labels
    tmp = np.array([])
    for text in text_array:
        text = text.strip()
        pattern = re.compile(r'</?\w+[^>]*>', re.S)
        text = pattern.sub('',text)
        text = text.replace('%', '')
        tmp = np.append(tmp, text)
        #print(text)  # debug
    return tmp

def filter_stopwords(text_array):
    # task 4: filter stopwords 
    tmp = []
    stop_words = set(stopwords.words('english'))
    for text in text_array:
        text_tk = [t.lower() for t in text if len(t.lower()) > 1 and t.lower() not in stop_words]
        tmp.append(text_tk)
    return tmp

def part_of_speech(text_array, nlp):
    # task 5 part of speech to each text
    tmp = []
    for text in text_array:
        tmp.append(nlp.pos_tag(text))
    return tmp  

def count_pattern(text_array, ddict):
    for words in text_array:
        if len(words) == 1:
            continue
        else:
            for i in range(len(words)):
                for j in range(i+1, len(words)):
                    if words[i] + ' ' + words[j] in ddict.keys():
                        ddict[words[i] + ' ' + words[j]]+=1
                    elif words[j] + ' ' + words[i] in ddict.keys():
                        ddict[words[j] + ' ' + words[i]]+=1
                    else:
                        ddict[words[i] + ' ' + words[j]]=1
    return ddict

if __name__ == "__main__":

    nlp = StanfordCoreNLP(r'C:\\Code\\stanford-corenlp-full-2018-10-05')
    ta = get_text_field('14res_data_sampled.jsonl')
    defaultdict = {}

    ta = preprocess(ta)
    words = [word_tokenize(text) for text in ta]
    text_tk = filter_stopwords(words) 
    pos = part_of_speech(ta, nlp)
    defaultdict = count_pattern(text_tk, defaultdict)
    defaultdict = {k:v for k,v in sorted(defaultdict.items(), key = lambda x: x[1], reverse=True)}
    #print_dict(defaultdict) # debug
    dict2file(defaultdict, "print") # save to txt
