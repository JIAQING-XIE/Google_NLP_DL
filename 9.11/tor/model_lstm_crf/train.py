from process_lap_data import ProcessData,Evaluate
from lstm_crf import BiLSTM_CRF
import torch
from config import  *
import torch.nn as nn
import numpy as np
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
np.random.seed(np_seed)


if __name__=="__main__":
    process_utils = ProcessData()
    process_utils.read_vocab(
        "C:\\Users\\11415\\Desktop\\Google_Deep_Learning\\Google_NLP_DL\\9.11\\tor\\vocab\\word_vocab.txt")
    test_texts,test_words,test_labels = process_utils.read_data( \
        "C:\\Users\\11415\\Desktop\\Google_Deep_Learning\\Google_NLP_DL\\9.11\\tor\\data_plain\\test.txt")
    test_words_ids, test_labels_ids=  process_utils.convert_to_vocab(test_words,test_labels)

    texts,words,labels = process_utils.read_data( \
        "C:\\Users\\11415\\Desktop\\Google_Deep_Learning\\Google_NLP_DL\\9.11\\tor\\data_plain\\train.txt")
    words_ids, labels_ids = process_utils.convert_to_vocab(words,labels)
    datas = [[texts[i],words[i],words_ids[i],labels_ids[i]] for i in range(len(texts))]
    test_datas = [[test_texts[i],test_words[i],test_words_ids[i],test_labels_ids[i]] for i in range(len(test_texts))]
    print(test_datas[0])
    val_sample_ids = np.random.choice(len(datas),  int(len(datas) * 0.1), replace=False)
    train_datas,train_labels = [],[]
    dev_datas,dev_labels = [],[]
    for i,data in enumerate(datas):
        if i in val_sample_ids:
            dev_datas.append(data)
            dev_labels.append(labels_ids[i])
        else:
            train_datas.append(data)
            train_labels.append(labels_ids[i])
    print("train_dev_splited..")
    word_embedding_matrix = process_utils.loadEmbMatrix( \
        "C:\\Users\\11415\\Desktop\\Google_Deep_Learning\\Google_NLP_DL\\9.11\\tor\\data_plain\\aets_embedding.txt", embedding_size, bina=False)
    print("embedding..")
    model = BiLSTM_CRF(len(word_embedding_matrix),labels_dict,embedding_size,lstm_hidding_dim*2,word_embedding_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_epoch_loss = 0.0
    train_batches = process_utils.batch_iter(train_datas, train_labels, batch_size, embedding_size)
    each_epoch_batch_number = int(len(train_datas) / batch_size)
    evalute_utils = Evaluate()
    dev_max_f1 = 0.0
    for k, train_batch in enumerate(train_batches):
        model.train()
        train_texts_batch, train_words_batch, train_words_ids_batch, train_aspect_labels_batch = zip(*train_batch)
        train_aspect_labels_batch = np.array(train_aspect_labels_batch)
        x_train = torch.from_numpy(np.array(train_words_ids_batch)).long()
        train_real_lengths_batch = [len([_ for _ in tw if _!='unk']) for tw in train_words_batch]
        train_real_lengths_batch_mask = [[[1] for i in range(_)]+[[0] for i in range(max_length-_)] for _ in train_real_lengths_batch]
        train_real_lengths_batch_mask = torch.from_numpy(np.array(train_real_lengths_batch_mask))
        optimizer.zero_grad()
        sample_loss = model.neg_log_likelihood(x_train,torch.from_numpy(train_aspect_labels_batch),train_real_lengths_batch,train_real_lengths_batch_mask)
        sample_loss.backward()
        optimizer.step()
        train_epoch_loss += sample_loss.item()
        if (k + 1) % each_epoch_batch_number == 0 or k==0:
            model.adjust_learning_rate(learning_rate, optimizer, int((k + 1) / each_epoch_batch_number ))
            print(int((k + 1) / each_epoch_batch_number )," epoch , train_loss:", train_epoch_loss)
            train_epoch_loss = 0.0
            model.eval()
            dev_batches = process_utils.batch_iter(dev_datas, dev_labels, 500, 1, False)
            dev_epoch_reals,dev_epoch_predict,dev_epoch_words = [],[],[]
            for m, dev_batch in enumerate(dev_batches):
                dev_texts_batch, dev_words_batch, dev_words_ids_batch, dev_aspect_labels_batch = zip(*dev_batch)
                dev_aspect_labels_batch = np.array(dev_aspect_labels_batch)
                x_dev = torch.from_numpy(np.array(dev_words_ids_batch)).long()
                dev_real_lengths_batch = [len([_ for _ in tw if _ != 'unk']) for tw in dev_words_batch]
                predict_sample = model.predict(x_dev,dev_real_lengths_batch)
                dev_epoch_reals.extend(dev_aspect_labels_batch)
                dev_epoch_predict.extend(predict_sample)
                dev_epoch_words.extend(dev_words_batch)
            dev_precison, dev_recall, dev_f1 = evalute_utils.calculate(dev_epoch_words, dev_epoch_predict,dev_epoch_reals,)
            if dev_f1>dev_max_f1:
                dev_max_f1 = dev_f1
                print("dev evaluating...", dev_precison, dev_recall, dev_f1)
                #######test
                model.eval()
                test_batches = process_utils.batch_iter(test_datas, test_labels, 500, 1, False)
                test_epoch_reals, test_epoch_predict, test_epoch_words = [], [], []
                for p, test_batch in enumerate(test_batches):
                    test_texts_batch, test_words_batch, test_words_ids_batch, test_aspect_labels_batch = zip(*test_batch)
                    test_aspect_labels_batch = np.array(test_aspect_labels_batch)
                    test_real_lengths_batch = [len([_ for _ in tw if _ != 'unk']) for tw in test_words_batch]
                    x_test = torch.from_numpy(np.array(test_words_ids_batch)).long()
                    predict_sample = model.predict(x_test,test_real_lengths_batch)
                    test_epoch_reals.extend(test_aspect_labels_batch)
                    test_epoch_predict.extend(predict_sample)
                    test_epoch_words.extend(test_words_batch)
                test_precison, test_recall, test_f1 = evalute_utils.calculate(test_epoch_words,test_epoch_predict, test_epoch_reals,
                                                                           )
                print("test evaluating...",test_precison, test_recall, test_f1)
                
