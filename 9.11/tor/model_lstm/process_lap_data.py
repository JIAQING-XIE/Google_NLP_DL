from nltk import word_tokenize
from tor.model_lstm.lstm import *
from gensim.models.keyedvectors import KeyedVectors
import numpy as np


class ProcessData():

    def __init__(self):
        self.max_length = 83

    def process_text(delf,text):
        dot_exist = ('.' in text)
        cur_text = text.replace('.', '')
        # cur_text = cur_text.replace('-', ' ')
        cur_text = cur_text.replace(' - ', ', ').strip()
        cur_text = cur_text.replace('- ', ' ').strip()
        # split words and punctuations
        if '? ' not in cur_text:
            cur_text = cur_text.replace('?', '? ').strip()
        if '! ' not in cur_text:
            cur_text = cur_text.replace('!', '! ').strip()
        cur_text = cur_text.replace('(', '')
        cur_text = cur_text.replace(')', '')
        cur_text = cur_text.replace('...', ', ').strip('.').strip().strip(',')
        # remove quote
        cur_text = cur_text.replace('"', '')
        cur_text = cur_text.replace(" '", " ")
        cur_text = cur_text.replace("' ", " ")

        cur_text = cur_text.replace(':', ', ')
        if dot_exist:
            cur_text += '.'
            # correct some typos
        # mainly for processing English texts
        cur_text = cur_text.replace('cant', "can't")
        cur_text = cur_text.replace('wouldnt', "wouldn't")
        cur_text = cur_text.replace('dont', "don't")
        cur_text = cur_text.replace('didnt', "didn't")
        cur_text = cur_text.replace("you 're", "you're")

        # replace some special symbol
        cur_text = cur_text.replace(u' – ', ', ').strip()

        cur_text = cur_text.replace(u"‘", "")
        # filter the non-ascii character
        cur_text = ''.join([ch if ord(ch) < 128 else ' ' for ch in cur_text])
        return cur_text

    def read_data(self,path):
        all_labels,all_texts,all_words = [],[],[]
        entity_number = {'POS':0,'NEG':0,'NEU':0}
        with open(path,encoding='utf-8') as reader:
            for i,line in enumerate(reader.readlines()):
                cur_text = line.strip().split("####")[0].lower()
                cur_text = self.process_text(cur_text)
                words = word_tokenize(cur_text)
                labels_plain = [_.split('=')[1] for _ in line.strip().split("####")[1].split(' ')]
                w_plain = [_.split('=')[0] for _ in line.strip().split("####")[1].split(' ')]

                ####转labels为bieso模式
                labels = []
                for j,label in enumerate(labels_plain):
                    if label in ['T-POS','T-NEG','T-NEU']:
                        if j==0:
                            if labels_plain[j+1]=='O':
                                labels.append('S-'+label.split('-')[1])
                            else :
                                labels.append('B-'+label.split('-')[1])

                        elif j==len(labels_plain)-1:
                            if labels_plain[j-1]=='O':
                                labels.append('S-' + label.split('-')[1])
                            else:
                                labels.append('E-' + label.split('-')[1])
                        else:
                            if labels_plain[j-1]=='O' and labels_plain[j+1]=='O':
                                labels.append('S-' + label.split('-')[1])
                            elif labels_plain[j-1]=='O' and labels_plain[j+1]!='O':
                                labels.append('B-' + label.split('-')[1])
                            elif labels_plain[j - 1] != 'O' and labels_plain[j + 1] != 'O':
                                labels.append('I-' + label.split('-')[1])
                            elif labels_plain[j - 1] != 'O' and labels_plain[j + 1] == 'O':
                                labels.append('E-' + label.split('-')[1])
                    else:
                        labels.append('O')
                ##统计分布
                for lab in labels:
                    if lab.split('-')[0] in ['B','S']:
                        entity_number[lab.split('-')[1]] += 1

                all_texts.append(cur_text)
                all_words.append(self.max_length_text(words))
                all_labels.append(labels)
        print(entity_number)
        return all_texts,all_words,all_labels

    def max_length_label(self,text):
        if len(text)>=self.max_length:
            return text[:self.max_length]
        else:
            return text+[0 for i in range(self.max_length-len(text))]

    def max_length_text(self,text):
        if len(text)>=self.max_length:
            return text[:self.max_length]
        else:
            return text+['unk' for i in range(self.max_length-len(text))]

    def read_vocab(self,path):
        self.vocab_word_to_id = {}
        self.vocab_id_to_word = {}
        with open(path,encoding='utf-8') as reader:
            for line in reader.readlines():
                self.vocab_id_to_word[len(self.vocab_word_to_id)] = line.strip()
                self.vocab_word_to_id[line.strip()] = len(self.vocab_word_to_id)

    def convert_to_vocab(self,texts,labels):
        texts_ids = []
        for text in texts:
            current_text_ids = []
            for tk in text:
                try:
                    current_text_ids.append(self.vocab_word_to_id[tk])
                except:
                    current_text_ids.append(0)
            texts_ids.append(current_text_ids)
        labels_ids = []
        for label in labels:
            labels_ids.append(self.max_length_label([labels_dict[_] for _ in label]))

        return texts_ids,labels_ids

    def loadEmbMatrix(self,pretrain_emb_path,embed_size, bina):

        print('Indexing word vectors.')
        model = KeyedVectors.load_word2vec_format(pretrain_emb_path, binary=bina)
        embedding_matrix = np.zeros((len(self.vocab_word_to_id), embed_size), dtype='float32')
        for i, tt in enumerate(list(self.vocab_word_to_id.keys())):
            if i % 30000 == 0:
                print(i)
            try:
                embedding_matrix[i] = model[tt]
            except KeyError:
                embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embed_size)
        return embedding_matrix

    def batch_iter(self,data,labels, batch_size, num_epochs, shuffle=True):

        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

class Evaluate():

    def calculate(self,texts,predicts,reals):
        real_number,predict_number,correct_number = 0,0,0

        for i,real in enumerate(reals):
            real_aspepct = []
            predict_aspepct = []
            current_predict = predicts[i]
            real_length = len([_ for _ in texts[i] if _!='unk'])
            real = real[:real_length]
            current_predict = current_predict[:real_length]
            for j,r1 in enumerate(real):
                if r1 in [1,5,9]:  #b
                    if j == len(real) - 1:
                        continue
                    else:
                        words_length = 0
                        if_e_exited = False
                        for r2 in real[j + 1:]:
                            if r2 == r1 + 1:
                                words_length += 1
                            elif r2 == r1 + 2:
                                words_length += 1
                                if_e_exited = True
                                break
                            else:
                                break
                        if if_e_exited:#如果有e结尾
                            real_aspepct.append((j, j + words_length, texts[i][j:j + 1 + words_length],sentiments_dict[r1]))
                elif r1 in [4,8,12]:#s
                    real_aspepct.append((j, j,texts[i][j],sentiments_dict[r1]))
            for j,p1 in enumerate(current_predict):
                if p1 in [1,5,9]:  #b
                    if j == len(current_predict) - 1:
                        continue
                    else:
                        words_length = 0
                        if_e_exited = False
                        for p2 in current_predict[j + 1:]:
                            if p2 == p1 + 1:
                                words_length += 1
                            elif p2 == p1 + 2:
                                words_length += 1
                                if_e_exited = True
                                break
                            else:
                                break
                        if if_e_exited:#如果有e结尾
                            predict_aspepct.append((j, j + words_length, texts[i][j:j + 1 + words_length],sentiments_dict[p1]))
                elif p1 in [4,8,12]:#b
                    predict_aspepct.append((j, j,texts[i][j],sentiments_dict[p1]))

            real_number += len(real_aspepct)
            predict_number += len(predict_aspepct)

            for po in predict_aspepct:
                if po in real_aspepct:
                    correct_number += 1

        if predict_number != 0:
            precison = correct_number / predict_number
        else:
            precison = 0
        if real_number != 0:
            recall = correct_number / real_number
        else:
            recall = 0
        if precison + recall != 0.0:
            f1 = 2 * precison * recall / (precison + recall)
        else:
            f1 = 0
        #print(real_number,predict_number,correct_number)
        return precison,recall,f1

