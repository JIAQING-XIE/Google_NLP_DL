from tf.model_lstm_crf.process_lap_data import ProcessData
from tf.model_lstm_crf.lstm_crf import  LSTM
import tensorflow as tf
import numpy as np
from tf.model_lstm_crf.config import *


if __name__=="__main__":
    process_utils = ProcessData()
    process_utils.read_vocab("../vocab/word_vocab.txt")
    test_texts, test_words, test_labels = process_utils.read_data("../data_plain/test.txt")
    test_words_ids, test_labels_ids = process_utils.convert_to_vocab(test_words, test_labels)
    # print(len(test_labels_ids))

    texts, words, labels = process_utils.read_data("../data_plain/train.txt")
    words_ids, labels_ids = process_utils.convert_to_vocab(words, labels)
    datas = [[texts[i], words[i], words_ids[i], labels_ids[i]] for i in range(len(texts))]
    test_datas = [[test_texts[i], test_words[i], test_words_ids[i], test_labels_ids[i]] for i in range(len(test_texts))]
    # 271,599,851,1007,1281,1811,2917,3961,4485,4522
    # 17436,48766,51180,63086
    # 17436,30244,48766
    np.random.seed(7894)
    val_sample_ids = np.random.choice(len(datas), int(len(datas) * 0.1), replace=False)
    train_datas, train_labels = [], []
    dev_datas, dev_labels = [], []
    for i, dd in enumerate(datas):
        if i in val_sample_ids:
            dev_datas.append(dd)
            dev_labels.append(labels_ids[i])
        else:
            train_datas.append(dd)
            train_labels.append(labels_ids[i])

    print("train_dev_splited..")

    word_embedding_matrix = process_utils.loadEmbMatrix("../data_plain/aets_embedding.txt", 300, bina=False)
    print("embedding..")

    mymodel = LSTM(300, word_embedding_matrix, process_utils.max_length)
    mymodel.network()
    n_iter = tf.Variable(0)
    each_epoch_batch_number = int(len(train_datas) / batch_size)
    learning_rate = tf.train.inverse_time_decay(learning_rate=learning_rate,
                       global_step=n_iter,
                       decay_steps=each_epoch_batch_number,
                       decay_rate=decay_rate,)
   # learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=n_iter, decay_steps=each_epoch_batch_number, decay_rate=0.95, staircase=False)
    # 启动模型
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.9).minimize(mymodel.loss,global_step=n_iter)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)

    print("加载完备")
    loss_epoch = 0.0
    acc_epoch = 0.0

    train_batches = process_utils.batch_iter(train_datas, train_labels, batch_size, 300)
    train_aspect_results, train_opinion_results, train_aspect_reals, train_opinion_reals, train_texts_result = [], [], [], [], []
    train_loss_epoch = 0.0
    max_dev_f1 = 0.0
    lr = 0.0
    train_t_loss_epoch, train_ts_loss_epoch, train_tg_loss_epoch, train_opt_loss_epoch = 0.0, 0.0, 0.0, 0.0
    for k, train_batch in enumerate(train_batches):
        n_iter = k
        train_texts_batch, train_words_batch, train_words_ids_batch, train_aspect_labels_batch = zip(*train_batch)
        train_lengths_batch = [len([_ for _ in tt if _!='unk']) for tt in train_words_batch]
        _, train_loss_batch, crf_logit,crf_matrix = \
            sess.run([train_op, mymodel.loss,mymodel.y_t_predict, mymodel.transition_params],
                     feed_dict={
                         mymodel.dropout: 0.5,
                         mymodel.input_text_id: train_words_ids_batch,
                         mymodel.input_aspect_labels: train_aspect_labels_batch,
                         mymodel.real_length:train_lengths_batch
                     })

        train_loss_epoch += train_loss_batch

        if (k+1) % each_epoch_batch_number == 0 and k != 0:
           # n_iter = int(k / each_epoch_batch_number)
            print("train_loss:", train_loss_epoch)
           # print(crf_matrix)
            train_loss_epoch = 0.0
            dev_batches = process_utils.batch_iter(dev_datas, dev_labels, 500, 1, False)
            dev_predict_labels = []
            dev_words_real = []
            for m, dev_batch in enumerate(dev_batches):
                dev_texts_batch, dev_words_batch, dev_words_ids_batch, dev_aspect_labels_batch = zip(*dev_batch)
                dev_lengths_batch = [len([_ for _ in tt if _ != 'unk']) for tt in dev_words_batch]
                dev_aspect_results_batch = []
                lr,dev_loss_batch, crf_logit,crf_matrix,decode_tags = \
                    sess.run([learning_rate,mymodel.loss, mymodel.y_t_predict, mymodel.transition_params,mymodel.decode_tags],
                             feed_dict={
                                 mymodel.dropout : 1.0,
                                 mymodel.input_text_id: dev_words_ids_batch,
                                 mymodel.input_aspect_labels: dev_aspect_labels_batch,
                                 mymodel.real_length: dev_lengths_batch
                             })
                #for o, cur_logits in enumerate(crf_logit):
                   # dev_aspect_results_batch.append(tf.contrib.crf.viterbi_decode(
                      #  cur_logits, crf_matrix)[0])
                dev_words_real.extend(dev_words_batch)
                dev_predict_labels.extend(decode_tags)
            dev_precison, dev_recall, dev_f1 = process_utils.caculate(dev_words_real, dev_predict_labels, dev_labels)
            print("epoch", int(k / each_epoch_batch_number), lr)
            if dev_f1 > max_dev_f1:
                #######test
                max_dev_f1 = dev_f1
                test_batches = process_utils.batch_iter(test_datas, test_labels_ids, 500, 1, False)
                test_predict_labels = []
                test_words_real = []
                for m, test_batch in enumerate(test_batches):
                    test_texts_batch, test_words_batch, test_words_ids_batch, test_aspect_labels_batch = zip(
                        *test_batch)
                    test_aspect_results_batch = []
                    test_lengths_batch = [len([_ for _ in tt if _ != 'unk']) for tt in test_words_batch]
                    lr,test_loss_batch,crf_logit,crf_matrix,decode_tags = \
                        sess.run([learning_rate,mymodel.loss,  mymodel.y_t_predict, mymodel.transition_params,mymodel.decode_tags],
                                 feed_dict={
                                     mymodel.dropout: 1.0,
                                     mymodel.input_text_id: test_words_ids_batch,
                                     mymodel.input_aspect_labels: test_aspect_labels_batch,
                                     mymodel.real_length:test_lengths_batch
                                 })
                    test_predict_labels.extend(decode_tags)
                    test_words_real.extend(test_words_batch)
                test_precison, test_recall, test_f1 = process_utils.caculate(test_words_real, test_predict_labels, test_labels_ids,True)
                #print(crf_matrix)
                print("dev precison,recall,f1 :", dev_precison, dev_recall, dev_f1)
                print("test precison,recall,f1 :", test_precison, test_recall, test_f1)
