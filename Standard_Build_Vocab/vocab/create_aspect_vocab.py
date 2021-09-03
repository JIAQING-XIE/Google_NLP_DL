import re
import jsonlines
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from stanfordcorenlp import StanfordCoreNLP
path  = r'C:\\Code\\stanford-corenlp-full-2018-10-05'
nlp = StanfordCoreNLP(path)

class ProcessData():
    def __init__(self,path):
        self.path = path

    def read_data(self):
        ##### 1、删除前后空格，小写 2、正则去除html符号,标点符号 3、文本分词 4、去除停用词
        self.datas = []
        pattern = re.compile(r'<[^>]+>', re.S)
        punc = '！@#￥%&*（）—“：’；、。，？》《%'
        english_stop_words = stopwords.words('english')+['the','and','is','a']
        with open(self.path,encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                data = pattern.sub('', item['text'].strip().lower())
                data = re.sub(r"[%s]+" % punc, "", data)
                self.datas.append(data)
        #分词:nltk/re
        self.datas_tk  = [word_tokenize(da) for da in self.datas]
        # 去除停用词
        self.datas_tk_new = []
        for data_tk in self.datas_tk:
            self.datas_tk_new.append([_ for _ in data_tk if _ not in english_stop_words])

    def get_cooccurrence_matrix(self):
        cooccurrence_matrix = defaultdict(int)
        for o,data_tk in enumerate(self.datas_tk_new):
            pos = nlp.pos_tag(self.datas[o])
            for i,word_tk_1 in enumerate(data_tk):
                for j,word_tk_2 in enumerate(data_tk):
                    if i<j:
                        if pos[i][1] in ['JJ','JJR','JJS','IN','DT'] or pos[j][1] in ['NN','NNS','NNP','NNPS']:
                            cooccurrence_matrix[(word_tk_2, word_tk_1)] += 1
                        elif pos[j][1] in ['JJ','JJR','JJS','IN','DT'] or pos[i][1] in ['NN','NNS','NNP','NNPS']:
                            cooccurrence_matrix[(word_tk_1, word_tk_2)] += 1

        cooccurrence_matrix_sorted = sorted(cooccurrence_matrix.items(),key=lambda kv: (kv[1], kv[0]),reverse=True)
        with open("cooccurrence_matrix.txt", 'w', encoding='utf-8') as writer:
            for co in cooccurrence_matrix_sorted:
                writer.write(' '.join(co[0])+'\t'+str(co[1])+'\n')

        nlp.close()

    def synonym_mining(self):
        # 利用nltk/word2vec
        current_aspects = []
        with open("aspect_vocab.txt", encoding='utf-8') as reader:
            for line in reader.readlines():
                current_aspects.extend(line.strip().split(','))
        #####nltk
        '''
        for aspect in current_aspects:
            synonyms =[]
            for syn in wordnet.synsets(aspect):
                for lm in syn.lemmas():
                    synonyms.append(lm.name())
           # print(aspect,synonyms)
        '''
        #####word2vec
        emb_model = KeyedVectors.load_word2vec_format("../../GoogleNews-vectors-negative300.bin",binary=True)
        for aspect in current_aspects:
            synoyms = emb_model.wv.similar_by_word(aspect)
            #print(aa,synoyms)

if __name__ == "__main__":
    pd = ProcessData("C:\\Users\\11415\\Desktop\\Google Deep Learning\\Standard_Build_Vocab\\vocab\data\\14res_data_sampled.jsonl")
    pd.read_data()
    pd.get_cooccurrence_matrix()
   # pd.synonym_mining()
