import tensorflow as tf
from tf.model_lstm.config import *
tf.set_random_seed(tf_seed)

class LSTM():
    def __init__(self,embedding_size,word_embedding_matrix,max_length):
        self.input_text_id = tf.placeholder(tf.int32, [None, max_length], name="input_text_id")
        self.input_aspect_labels = tf.placeholder(tf.int32, [None, max_length], name="input_aspect_labels")
        self.word_embedding_matrix = word_embedding_matrix
        self.dropout = tf.placeholder(tf.float32, None, name="drop")
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.real_length = tf.placeholder(tf.int32, [None], name="real_length")

    def network(self):
        with tf.device('/cpu:0'), tf.name_scope("matrix"):
            ##初始化词向量，保证unk向量一直为0
            word_maxtrix = tf.Variable(initial_value=self.word_embedding_matrix, name='word_embedding_matrix',
                                       dtype=tf.float32, trainable = True)
            self.word_embedding = tf.nn.embedding_lookup(word_maxtrix, self.input_text_id)
            self.word_embedding = tf.nn.dropout(self.word_embedding,keep_prob=self.dropout)

        with tf.device('/cpu:0'), tf.name_scope("aspect"):
            with tf.name_scope("lstm_t"):
                lstm_fw_cell_t = tf.nn.rnn_cell.BasicRNNCell(50,name="cell_fw_t")
                lstm_bw_cell_t = tf.nn.rnn_cell.BasicRNNCell(50,name="cell_bw_t")
                (outputs_t, output_states_t) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_t, lstm_bw_cell_t,
                                                                           self.word_embedding,sequence_length=self.real_length, dtype=tf.float32)
                word_lstm_embedding_t = tf.concat(outputs_t, 2)  # b,100,300

                ##lstm+cross-loss
                y_t_predict = tf.layers.dense(word_lstm_embedding_t, 13)

                y_t_predict_softmax = tf.nn.softmax(y_t_predict, name='Y_T_softmax')
                self.y_t_predict_label = tf.argmax(y_t_predict_softmax, -1, name='Y_T_label')
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_t_predict,
                                                                                   labels=tf.one_hot(self.input_aspect_labels,13)))











