import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    # cascaded LSTMs for joint aspect detection and sentiment prediction
    def __init__(self, params, vocab, embeddings, char_embeddings):
        super(Model, self).__init__()
        """
        :param params:
        :param vocab:
        :param embeddings:
        :param char_embeddings:
        """
        self.params = params
        self.name = 'lstm_cascade'
        self.dim_char = params.dim_char
        self.dim_w = params.dim_w
        self.dim_char_h = params.dim_char_h
        self.dim_ote_h = params.dim_ote_h
        self.dim_ts_h = params.dim_ts_h
        self.input_win = params.input_win
        self.ds_name = params.ds_name
        # tag vocabulary of opinion target extraction and targeted sentiment
        self.ote_tag_vocab = params.ote_tag_vocab
        self.ts_tag_vocab = params.ts_tag_vocab
        self.dim_ote_y = len(self.ote_tag_vocab)
        self.dim_ts_y = len(self.ts_tag_vocab)
        self.n_epoch = params.n_epoch
        self.dropout_rate = params.dropout
        self.tagging_schema = params.tagging_schema
        self.clip_grad = params.clip_grad
        self.use_char = params.use_char
        # name of word embeddings
        self.emb_name = params.emb_name
        self.embeddings = embeddings
        self.vocab = vocab
        # character vocabulary
        self.char_vocab = params.char_vocab
        #self.td_proportions = params.td_proportions
        self.epsilon = params.epsilon
        #self.tc_proportions = params.tc_proportions
        self.pc = dy.ParameterCollection()