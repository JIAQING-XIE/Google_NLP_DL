import torch
import torch.nn as nn
import torch.nn.functional as F
from config import  *

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp_bacth(vec):
    max_score_vec = torch.max(vec, dim=1)[0]
    max_score_broadcast = max_score_vec.view(vec.shape[0], -1).expand(vec.shape[0], vec.size()[1])
    return max_score_vec + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,word_embedding_matrix):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)     # 加上beg,end size=15
        self.dropout = nn.Dropout(p=dropout)

        word_embedding_matrix = torch.from_numpy(word_embedding_matrix)
        self.word_embeds = nn.Embedding.from_pretrained(word_embedding_matrix)
        self.word_embeds.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # 通过线性层得到每一个tag的分类概率，为发射矩阵，大小为b,l,15
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 转移概率矩阵.此处T[i][j]代表的是从tag-j到tag-i的一个概率分布
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 从任何词到begin是不可能的，从end到任意词的概率也是负无穷
        self.transitions.data[tag_to_ix["BEGIN"], :] = -10000
        self.transitions.data[:, tag_to_ix["END"]] = -10000

        self.hidden = self.init_hidden()#LSTM状态层

    def init_hidden(self, bacth=1):
        return (torch.randn(2, bacth, self.hidden_dim // 2),
                torch.randn(2, bacth, self.hidden_dim // 2))

    def _forward_alg_parallel(self, feats,real_lengths):
        # 前向算法，计算每一个时刻到下一个时刻每一个tag的概率分数。假设有K个标签
        # 从时刻i到时刻i+1，有k*k条路径概率，每个时刻只保留时刻i到时刻i+1的tagi+1可能性最大的一条路径。
        # for i in length:
        # step1:i+1时刻，tagi+1的分数 = max(forward分数 + tagk到下一个tagi+1的分数 + 发射到tagi+1的概率)  k条路径取最大,然后存在alphas_t中
        # 遍历完后，alphas_t中存的就是最后一个节点取tag1-k的的一个分数，即K条路径
        # alphas_t取最大路径作为最后结果。
        init_alphas = torch.full((feats.shape[0], self.tagset_size), -10000.)
        init_alphas[:, self.tag_to_ix["BEGIN"]] = 0.
        forward_var = init_alphas
        convert_feats = feats.permute(1, 0, 2)  # 从b,l,15到l,b,15，完成crf的Batch计算
        # mask
       # print(real_lengths[0])
        real_lengths = real_lengths.permute(1,0,2)#l,b

        for i,feat in enumerate(convert_feats):  # 每一个时刻做前向，长度为seq_length
            alphas_t = []  # 保存在每一个时刻到下一个时刻每一个tag的最大路径分数
            for next_tag in range(self.tagset_size):
                emit_score = feat[:, next_tag].view(
                    feats.shape[0], -1).expand(feats.shape[0], self.tagset_size)
                emit_score = emit_score*real_lengths[i]  #mask

                trans_score = self.transitions[next_tag].view(1, -1).repeat(feats.shape[0], 1)
                trans_score = trans_score*real_lengths[i]
                #tagi+1的分数 = max(forward分数 + tagk到下一个tagi+1的分数 + 发射到tagi+1的概率)  k条路径取最大,然后存在alphas_t中
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp_bacth(next_tag_var))
            forward_var = torch.stack(alphas_t).permute(1, 0)
        terminal_var = forward_var + self.transitions[self.tag_to_ix["END"]].view(1, -1).repeat(feats.shape[0], 1)
        alpha = log_sum_exp_bacth(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden(bacth=len(sentence))
        embeds = self.word_embeds(sentence)
        embeds = self.dropout(embeds)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags,real_lengths):
        # 真实标签的分数score
        totalsocre_list = []
        seq = 0
        for feat, tag in zip(feats, tags): #batch，计算每一条样本
            totalscore = torch.zeros(1)
            tag = tag.long()
            tag = torch.cat([torch.tensor([self.tag_to_ix["BEGIN"]], dtype=torch.long), tag])
            for i, smallfeat in enumerate(feat):
                if i>=real_lengths[seq]: #mask
                    break
                totalscore = totalscore + \
                             self.transitions[tag[i + 1], tag[i]] + smallfeat[tag[i + 1]]
            totalscore = totalscore + self.transitions[self.tag_to_ix["END"], tag[real_lengths[seq]-1]]
            totalsocre_list.append(totalscore)
            seq+=1
        return torch.cat(totalsocre_list)

    def _viterbi_decode_predict(self, feats_list,real_lengths):
        # 解码viterbi算法，动态规划思想
        # for i in seq_length:
        # step1: 在时刻i计算到时刻i+1的k个tag的概率，
        path_list = []
        for j,feats in enumerate(feats_list): #对batch中的每一个样本计算 l,15
            backpointers = []
            # Initialize the viterbi variables in log space
            init_vvars = torch.full((1, self.tagset_size), -10000.)
            init_vvars[0][self.tag_to_ix["BEGIN"]] = 0
            forward_var = init_vvars
            for i,feat in enumerate(feats):# 对每个时刻计算.. feat 是当前时刻发射到tagk的概率
                if i>=real_lengths[j]:#mask
                    break
                bptrs_t = []
                viterbivars_t = []
                for next_tag in range(self.tagset_size):
                    # 时刻i到时刻i+1-tagk的路径有k条，选择分数最大的一条，然后保存下i时刻的tag_id
                    next_tag_var = forward_var + self.transitions[next_tag]##[1,15]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id) #存下i时刻的tag_id，用于回溯
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))#存下i时刻的分数
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)  #最大的分数路径加上发射概率分数，完成一个时刻t的迭代
                backpointers.append(bptrs_t)

            # 到end的概率
            terminal_var = forward_var + self.transitions[self.tag_to_ix["END"]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id] #从end开始回溯
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)

            start = best_path.pop()
            assert start == self.tag_to_ix["BEGIN"]  # Sanity check
            best_path.reverse()
            path_list.append(best_path)
        return path_list

    def neg_log_likelihood(self, sentence, tags,real_lengths,real_lengths_mask):
        feats = self._get_lstm_features(sentence)  # b,l,100 lstm embedding
        forward_score = self._forward_alg_parallel(feats,real_lengths_mask)  # 前向计算路径的分数
        gold_score = self._score_sentence(feats, tags,real_lengths)  # 计算真实标签的分
        return torch.mean(forward_score - gold_score)

    def predict(self, sentence,real_lengths):
        lstm_feats = self._get_lstm_features(sentence)
        tag_seq_list = self._viterbi_decode_predict(lstm_feats,real_lengths)
        return tag_seq_list

    def adjust_learning_rate(self,lr, optimizer, epoch,):
        lr = lr / (1 + epoch * decay_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr