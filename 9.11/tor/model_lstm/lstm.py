import torch
import torch.nn as nn
import torch.nn.functional as F
from tor.model_lstm.config import  *
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, word_embedding_matrix):
        super(LSTM, self).__init__()

       # self.word_emb = nn.Embedding(vocab_size, embedding_size)
       # self.word_emb.weight.data.copy_(torch.from_numpy(word_embedding_matrix))
       # self.word_emb.weight.requires_grad = False
        word_embedding_matrix = torch.from_numpy(word_embedding_matrix)
        self.word_emb = nn.Embedding.from_pretrained(word_embedding_matrix)
        self.word_emb.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=lstm_hidding_dim, bidirectional=True,batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

        self.classify_layer = nn.Linear(lstm_hidding_dim * 2, classfy_number)

    def forward(self, input_words_ids):
        input_words_ids_embedding= self.word_emb(input_words_ids)
        input_words_ids_embedding = self.dropout(input_words_ids_embedding)

        batch_size = input_words_ids_embedding.shape[0]

        hidden_state = torch.randn(1 * 2, batch_size,lstm_hidding_dim)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1 * 2, batch_size,lstm_hidding_dim)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        hidden_state = torch.nn.init.xavier_uniform_(hidden_state)
        cell_state = torch.nn.init.xavier_uniform_(cell_state)

        outputs, (_, _) = self.lstm(input_words_ids_embedding, (hidden_state, cell_state))

        scores = self.classify_layer(outputs)
        scores_softmax = F.softmax(scores,dim=-1)
        predict = torch.max(scores_softmax,dim=-1)[1]
        scores_log = F.log_softmax(scores,dim=-1)
        return scores_log,predict

    def adjust_learning_rate(self,lr, optimizer, epoch,):
        lr = lr / (1 + epoch * decay_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
