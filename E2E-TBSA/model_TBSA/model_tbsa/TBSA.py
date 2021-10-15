import torch
import torch.nn as nn
import torch.nn.functional as F
from config import  *
import random
import numpy as np
from torch.backends import cudnn
if torch.cuda.is_available():
    device = torch.device('cuda:0')
random.seed(seed)
np.random.seed(np_seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#if deterministic:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TBSA(nn.Module):
    def __init__(self, vocab_size, embedding_size, word_embedding_matrix,transition_scores):
        super(TBSA, self).__init__()


        self.word_emb = nn.Embedding.from_pretrained(word_embedding_matrix)
        self.word_emb.weight.requires_grad = False

        self.lstm1 = nn.LSTM(input_size=embedding_size, hidden_size=lstm_hidding_dim, bidirectional=True,batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_hidding_dim*2, hidden_size=lstm_hidding_dim, bidirectional=True,batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

        self.gt_layer = nn.Linear(lstm_hidding_dim*2,1,bias=True)
        self.classify_layer1 = nn.Linear(lstm_hidding_dim * 2, 5,bias=True)
        self.classify_layer2 = nn.Linear(lstm_hidding_dim * 2, classfy_number,bias=True)
        self.classify_layer3 = nn.Linear(lstm_hidding_dim * 2, 2,bias=True)  # oe
        self.transition_scores = transition_scores

    def lstm_embedding(self,input,real_batch_size):
        if torch.cuda.is_available():
            hidden_state1 = torch.randn(1 * 2, real_batch_size,lstm_hidding_dim).to(device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
            cell_state1 = torch.randn(1 * 2, real_batch_size,lstm_hidding_dim).to(device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        else:
            hidden_state1 = torch.randn(1 * 2, real_batch_size, lstm_hidding_dim)
            cell_state1 = torch.randn(1 * 2, real_batch_size, lstm_hidding_dim)
        hidden_state1 = torch.nn.init.xavier_uniform_(hidden_state1)
        cell_state1 = torch.nn.init.xavier_uniform_(cell_state1)

        outputs1, (_, _) = self.lstm1(input, (hidden_state1, cell_state1))
        if torch.cuda.is_available():
            hidden_state2 = torch.randn(1 * 2, real_batch_size,lstm_hidding_dim).to(device)
            cell_state2 = torch.randn(1 * 2, real_batch_size,lstm_hidding_dim).to(device)
        else:
            hidden_state2 = torch.randn(1 * 2, real_batch_size, lstm_hidding_dim)
            cell_state2 = torch.randn(1 * 2, real_batch_size, lstm_hidding_dim)
        hidden_state2 = torch.nn.init.xavier_uniform_(hidden_state2)
        cell_state2 = torch.nn.init.xavier_uniform_(cell_state2)
        outputs2, (_, _) = self.lstm2(outputs1, (hidden_state2, cell_state2))
        return outputs1,outputs2

    def get_loss(self,softmax_zs_last,softmax_zt,softmax_oe,opinion_labels,boundary_labels,oe_labels):
        ## 求交叉熵
        ## ts_loss
        softmax_zs_last_log = torch.log(softmax_zs_last)
        ts_loss = -torch.sum(torch.mean(torch.sum(torch.multiply(opinion_labels,softmax_zs_last_log),-1),-1))  ## mean(sum(b,l,13 * b,l,13)) == b
        # t_loss
        softmax_zt_log = torch.log(softmax_zt)
        t_loss = -torch.sum(torch.mean(torch.sum(torch.multiply(boundary_labels, softmax_zt_log), -1),-1))  ## mean(sum(b,l,13 * b,l,13)) == b

        ## oe_loss
        softmax_oe_log = torch.log(softmax_oe)
        oe_loss = -torch.sum(torch.mean(
            torch.sum(torch.multiply(oe_labels, softmax_oe_log), -1),-1))  ## mean(sum(b,l,13 * b,l,13)) == b

        all_loss = ts_loss + t_loss+ oe_loss

        return  all_loss

    def forward(self, input_words_ids,opinion_labels,boundary_labels,oe_labels):
        ##word-embedding
        input_words_ids_embedding= self.word_emb(input_words_ids)
        input_words_ids_embedding = self.dropout(input_words_ids_embedding)

        real_batch_size = input_words_ids_embedding.shape[0]
        lstm_ht,lstm_hs = self.lstm_embedding(input_words_ids_embedding,real_batch_size)

        # SC to get lstm_hs_2
        gt = self.gt_layer(lstm_ht)
        gt = F.sigmoid(gt.squeeze(dim=1))
        gt = gt.view(real_batch_size,max_length,1).repeat([1,1,lstm_hidding_dim*2])
        if torch.cuda.is_available():
            gt_plus = torch.ones([real_batch_size,max_length,lstm_hidding_dim*2]).to(device)-gt      ###1-gt
            lstm_hs_2 = torch.zeros([real_batch_size,max_length,lstm_hidding_dim*2]).to(device)
        else:
            gt_plus = torch.ones([real_batch_size, max_length, lstm_hidding_dim * 2]) - gt  ###1-gt
            lstm_hs_2 = torch.zeros([real_batch_size, max_length, lstm_hidding_dim * 2])
        for seq in range(max_length):
            if seq == 0:
                lstm_hs_2[:,seq,:] = lstm_hs[:,0,:]
            else:
                lstm_hs_2[:, seq, :] = gt[:,seq] * lstm_hs[:, seq, :] + gt_plus[:,seq]*lstm_hs_2[:,seq-1,:].clone()

        lstm_zt = self.classify_layer1(lstm_ht)  # bieos
        lstm_zs = self.classify_layer2(lstm_hs_2)  # b-pos,i-pos,......

        softmax_zt = F.softmax(lstm_zt,-1) # target B, I, O, E, S
        softmax_zs = F.softmax(lstm_zs,-1) # BG

        # BG
        softmax_zs_2 = torch.matmul(softmax_zt,self.transition_scores)   # b,l,5 * 5,13 = b,l,13
        ct_matrix = torch.matmul(softmax_zt,softmax_zt.permute([0,2,1]))   # b,l,5 * b,5,l = b,l,l
        if torch.cuda.is_available():
            ct = torch.zeros([real_batch_size,len(ct_matrix[0])]).to(device)  #b,l 代表每一个batch中每个节点的内积
        else:
            ct = torch.zeros([real_batch_size, len(ct_matrix[0])])
        for i in range(real_batch_size):
            ct[i] = torch.diag(ct_matrix[i])

        at = 0.5 * ct
        at = at.view(real_batch_size,len(ct[0]),1).repeat(1,1,13)
        softmax_zs_last = at * softmax_zs_2 + (1 - at) * softmax_zs

        # OE
        oe_scores = self.classify_layer3(lstm_ht)
        softmax_oe = F.softmax(oe_scores,-1)

        ## 交叉熵loss
        all_loss = self.get_loss(softmax_zs_last, softmax_zt,softmax_oe, opinion_labels, boundary_labels,oe_labels)

        predict = torch.max(softmax_zs_last,dim=-1)[1]
        return predict,all_loss

    def adjust_learning_rate(self,lr, optimizer, epoch,):
        lr = lr / (1 + epoch * decay_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
