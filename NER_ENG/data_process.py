import re
from stanfordcorenlp import StanfordCoreNLP
path  = r'C:\\Users\\11415\\Desktop\\stanford-corenlp-4.2.2\\'
nlp = StanfordCoreNLP(path)

class Data():
    def __init__(self, data_path, method = "BIOES"):
        self.data_path = data_path
        self.method = method

    def transform(self):
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
                        #print(type(sen_tk))
                        word_lists.append(sen_tk)
                        count+=1
                    else:
                        for element in part.split(' '):
                            tag_list.append(element.split("=")[1])
                            train_word_list.append(element.split("=")[0])
                train_word_lists.append(train_word_list)
                tag_lists.append(tag_list)
        
        #print(word_lists[:5])

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
                        



    def tokenize(self, sentence):
        """
        tokenize the sentence into an array of words with certain patterns
        """
        pattern = re.compile(r'<[^>]+>', re.S)
        punc = '！@#￥%&*（）—“：’；、。，？》《%'
        sentence = sentence.strip()
        sentence = pattern.sub('', sentence)
        sentence = re.sub(r"[%s]+" % punc, "", sentence)
        sen_tk = nlp.word_tokenize(sentence)
        return sen_tk

    def save(self):
        pass

    def train_test_split(self):
        """
        split to train and test datasets
        """
        pass