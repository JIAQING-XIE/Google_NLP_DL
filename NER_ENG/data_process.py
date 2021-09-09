import re
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from stanfordcorenlp import StanfordCoreNLP
path  = r'C:\\Users\\11415\\Desktop\\stanford-corenlp-4.2.2\\'
nlp = StanfordCoreNLP(path)

class Data():
    def __init__(self, data_path, method = "BIOES"):
        self.data_path = data_path
        self.method = method
    def transform(self, method = "re"):
        """
        1.分词
        2.transform T to B, I, E according to its position
        """
        word_lists = []
        train_word_lists = []
        tag_lists = []
        with open(self.data_path, encoding='utf-8') as reader:
            for line in reader.readlines():
                count = 0
                tag_list = []
                train_word_list = []
                for part in line.strip().split('####'): # take '####' as the split standard
                    if count == 0:
                        sen_tk = self.tokenize(part)
                        word_lists.append(sen_tk)
                        count+=1
                    else:
                        for element in part.split(' '):
                            if element[0] == "=":
                                tag_list.append(element.split("=")[2])
                                train_word_list.append("=")
                            else:
                                tag_list.append(element.split("=")[1])
                                train_word_list.append(element.split("=")[0])
                train_word_lists.append(train_word_list)
                tag_lists.append(tag_list)
        for i in range(len(train_word_lists)):
            assert train_word_lists[i] == word_lists[i], " line {}, \n {} \n {}".format(i, word_lists[i], train_word_lists[i])
        return train_word_lists, tag_lists
        
    def to_id(self, word_lists, tag_lists, make_vocab = False):
        if make_vocab:
            word2id = self.build_map(word_lists)
            tag2id = self.build_map(tag_lists)
            return word_lists, tag_lists, word2id, tag2id
        else:
            return word_lists, tag_lists

    def id2word(self, word2id):
        id_to_word = {id: word for (word, id) in word2id.items()}
        return id_to_word

    
    def extend_maps(self, word2id, tag2id, for_crf=True):
        word2id['<unk>'] = len(word2id)
        word2id['<pad>'] = len(word2id)
        tag2id['<unk>'] = len(tag2id)
        tag2id['<pad>'] = len(tag2id)
        # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
        if for_crf:
            word2id['<start>'] = len(word2id)
            word2id['<end>'] = len(word2id)
            tag2id['<start>'] = len(tag2id)
            tag2id['<end>'] = len(tag2id)

        return word2id, tag2id

    def check(self, group1, group2):
        """
        compare the array with the marked entities after '####' to check if they have the same length. 
        If not, delete that word in the tokenized group.
        """
        new_group = []
        i = 0
        j = 0
        while i < len(group1):
            while j < len(group2):
                if group1[i] == group2[j]:
                    new_group.append(group1[i])
                    i+=1 # move to next
                    j+=1 # move to next
                elif group1[i] in group2[j]: # meet mark behind
                    current = i
                    for current in range(i, len(group1) + 1):
                        if ''.join(group1[i:current]) in group2[j]:
                            if ''.join(group1[i:current]) == group2[j]:
                                new_group.append(group2[j])
                                i = current
                                j+=1
                                break
                        else:
                            break  
        return new_group
                        
    def substring(self, words):
        """
        check one string if its the substring of the other strings
        !!!! very slow and takes up a huge amount of CPU. Please do not use it.
        """
        for i in range(len(words)):
            for j in range(i, len(words)):
                if len(words) >= 4:
                    if words[i] in words[j]:
                        words[j] = words[i]
                    elif words[j] in words[i]:
                        words[i] = words[j]

        return words

    def statistics(self, tags, processed = False):
        for tag in tags:
            count = 0
            for i in range(len(tag)-1):
                if tag[i] != 'O' and not processed: # 一定是entity
                    if tag[i+1] == 'O' and count == 0:
                        tag[i] = 'S' + tag[i][1:]
                    elif tag[i+1] != 'O' and count==0:
                        tag[i] = 'B' + tag[i][1:]
                        tag[i+1] = 'I' + tag[i+1][1:]
                        count+=1
                    elif count!=0 and  tag[i+1] != 'O':
                        tag[i] = 'I'+ tag[i][1:]
                        tag[i+1] = 'E' + tag[i+1][1:]
                        count+=1
                    elif count!=0 and  tag[i+1] == 'O':
                        tag[i] = 'E' + tag[i][1:]
                        count = 0
                tmp = i + 1
                if tmp == len(tag) -1:
                    if tag[tmp] != 'O' and tag[i] == 'O':
                        tag[tmp] = 'S' + tag[tmp][1:]
                    break

        # count the number of entities:
        num_entities = {'POS':0, 'NEU':0, 'NEG':0}
        i = 0
        count = 0
        for ta in tags:
            while i < len(ta):
                if ta[i][:1] == 'S' or ta[i][:1] == 'E':
                    num_entities[ta[i][2:]]+=1
                i+=1
            i = 0
        print(num_entities)
        return tags
    
    def tokenize(self, sentence, method = "re"):
        """
        tokenize the sentence into an array of words with certain patterns
        """
        if method == "nlp":
            pattern = re.compile(r'<[^>]+>', re.S)
            punc = '！@#￥%&*（）—“：’；、。，？》《%'
            sentence = sentence.strip()
            sentence = pattern.sub('', sentence)
            sentence = re.sub(r"[%s]+" % punc, "", sentence)
            sen_tk = nlp.word_tokenize(sentence)
        elif method == "re":
            sentence = re.sub(r'(?<=\d)[\.](?=\d)','',sentence)
            sentence = re.sub(r' - ', ",", sentence)
            sentence = re.sub(r'-- ', " ", sentence)
            sentence = re.sub(r'- ', " ", sentence)
            sentence = re.sub(r'--', " -- ", sentence)
            sentence = sentence.replace(",", " , ")
            sentence = sentence.replace(":", " , ")
            sentence = sentence.replace(".", " . ")
            sentence = sentence.replace("=)", " = ")
            sentence = sentence.replace("?", " ? ")
            sentence = sentence.replace("!", " ! ")
            sentence = sentence.replace(";", " ; ")
            sentence = sentence.replace("$", " $ ")
            sentence = sentence.replace("%", " % ")
            sentence = sentence.replace("#", " # ")
            sentence = sentence.replace(">", " > ")
            #sentence = sentence.replace("*", " " )
            sentence = sentence.replace("n't", " n't")
            sentence = sentence.replace("'few", " few")
            sentence = re.sub(r'N\'T', " N'T",sentence)
            sentence = re.sub(r'didnt', "did n't",sentence)
            sentence = re.sub(r'dont', "do n't",sentence)
            sentence = re.sub(r'cant', "ca n't",sentence)
            sentence = re.sub(r'wouldnt', "would n't",sentence)
            sentence = re.sub(r'\'ve', " 've", sentence)
            sentence = re.sub(r'\'have', " have", sentence)
            sentence = re.sub(r'\'re', " 're", sentence)
            sentence = re.sub(r'\'s', " 's", sentence)
            sentence = re.sub(r'\'m ', " 'm ", sentence)
            sentence = re.sub(r'\'M ', " 'M ", sentence)
            sentence = re.sub(r'\'d', " 'd", sentence)
            sentence = re.sub(r'\'ll', " 'll", sentence)
            sentence = re.sub(r'[()]', "", sentence)
            sentence = re.sub(r'[\"\"]', "", sentence)
            sentence = re.sub(r'\' ', " ", sentence)
            sentence = re.sub(r'@', " @ ", sentence)
            sentence = sentence.replace("cannot", "can not")
            sentence = sentence.replace("  ", " ")
            sen_tk = re.split(" ", sentence)
            while '' in sen_tk:
                sen_tk.remove('') # 去除空字符
            i = 0
            while i < len(sen_tk) -1: # 去除多余句号(句中+句尾)
                if sen_tk[i] == "." and sen_tk[i+1] == ".":
                    del sen_tk[i]
                else:
                    i+=1
        return sen_tk

    def save(self):
        pass

    def train_valid_split(self, words, tags):
        """
        split to train and vadiation datasets
        X : words list->list
        y: tags list->list
        """
        X_train, X_valid, y_train, y_valid = train_test_split(words, tags, test_size=0.1, random_state=30244)
        print("----- Training statistics -----")
        train_tags = self.statistics(y_train, processed=True)
        print("----- Validation statistics -----")
        valid_tags = self.statistics(y_valid, processed=True)
        return X_train, X_valid, y_train, y_valid
    
    def build_map(self,lists):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps

class Glove():
    def __init__(self, vocab_size, embed_size, word2id):
        self.vocab_size =len(word2id)
        self.embed_size = embed_size
        self.word2id = word2id

    def glove_word2vec(self, glove_inputfile, word2vec_output_file):
        glove2word2vec(glove_inputfile, word2vec_output_file)
    
    def get_weight(self, file, word2id, id2word):
        wvmodel = KeyedVectors.load_word2vec_format(file \
            , binary=False, encoding='utf-8')
        torch.manual_seed(131415)
        weight = torch.Tensor(self.vocab_size, self.embed_size).uniform_(-0.5, 0.5)

        for i in range(len(wvmodel.index2word)):
            try:
                index = word2id[wvmodel.indexword[i]]
            except:
                continue
            weight[index, :] = torch.from_numpy(wvmodel.get_vector(
                id2word[word2id[wvmodel.index2word[i]]]))
        
        return weight
        
        


        
        