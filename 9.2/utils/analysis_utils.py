useful_pos_two_dict = {
    ('NN', 'JJ'):0, ('JJ', 'NN'):1, ('NN', 'JJR'):2, ('JJR', 'NN'):3, ('NN', 'JJS'):4, ('JJS', 'NN'):5,
    ('NNS', 'JJ'):6, ('JJ', 'NNS'):7, ('NNS', 'JJR'):8, ('JJR', 'NNS'):9,('NNS', 'JJS'):10,('JJS', 'NNS'):11,
    ('NNP', 'JJ'):12, ('JJ', 'NNP'):13, ('NNP', 'JJR'):14, ('JJR', 'NNP'):15, ('NNP', 'JJS'):16, ('JJS', 'NNP'):17,
    ('NNPS', 'JJ'):18, ('JJ', 'NNPS'):19, ('NNPS', 'JJR'):20, ('JJR', 'NNPS'):21, ('NNPS', 'JJS'):22, ('JJS', 'NNPS'):23,
    ('IN', 'JJ'):24, ('JJ', 'IN'):25, ('IN', 'JJR'):26, ('JJR', 'IN'):27, ('IN', 'JJS'):28,('JJS', 'IN'):29,
    ('DT', 'JJ'):30, ('JJ', 'DT'):31, ('DT', 'JJR'):32, ('JJR', 'DT'):33, ('DT', 'JJS'):34, ('JJS', 'DT'):35,
                  }

class AnalysisUtils:
    def get_single_sentence_lables(self,text_tk,aspect_vocab,dependency,pos):
        labels = []
        indexs = []
        for i, tk in enumerate(text_tk):
            if tk in aspect_vocab:##如果是aspect词典中的词
                for dep in dependency[1:]: #dep为连边 (连边类型,index1,index2)
                    if dep[1] == i + 1 or dep[2] == i + 1: #找到与当前词有连边的
                        pos_two = (pos[dep[1] - 1][1], pos[dep[2] - 1][1])
                        if pos_two in useful_pos_two_dict:
                            if dep[1] == i + 1: #保证是aspect+opinon的顺序
                                labels.append(text_tk[dep[1] - 1] + ' ' + text_tk[dep[2] - 1])
                            else:
                                labels.append(text_tk[dep[2] - 1] + ' ' + text_tk[dep[1] - 1])
                            current_index = [dep[1] - 1, dep[2] - 1]
                            current_index.sort()
                            indexs.append(current_index)
        return labels,indexs

    def analysis_paragraph(self,labels,indexs,text,text_tk):

        paragraphs = []
        for i,label in enumerate(labels):
            current_indexs = indexs[i]
            begin = current_indexs[0]
            end = current_indexs[1]
            paragraphs.append(' '.join(text_tk[begin:end+1]))
        return paragraphs

    def analysis_sentiment(self,labels,positive_vocab,negative_vocab):
        sentiments  = []
        for label in labels:
            opinion = label.split(' ')[1]
            if opinion in positive_vocab:
                sentiments.append(1)
            elif opinion in negative_vocab:
                sentiments.append(0)
            else:
               # print(label)
                sentiments.append(-1)
        return sentiments

    def analysis_deny(self,labels,indexs,dependency,text_tk,deny_vocab,sentiments):
        # 加上否定词分析
        for i, label in enumerate(labels):
            deny_number = 0
            current_index = indexs[i]
            for dep in dependency:
                if dep[1] - 1 in current_index or dep[2] - 1 in current_index:  # 找到与其有连边的词
                    if text_tk[dep[1] - 1] in deny_vocab or text_tk[dep[2] - 1] in deny_vocab:
                        deny_number += 1
            if deny_number % 2 == 1:
                labels[i] = label.split(' ')[0] + ' not ' + label.split(' ')[1]
                sentiments[i] = 1 - sentiments[i]
        return labels,sentiments


