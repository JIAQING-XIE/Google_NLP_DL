from process_lap_data import ProcessData,Evaluate
from lstm_crf import BiLSTM_CRF
import torch
from config import *
import torch.nn as nn
import numpy as np
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
np.random.seed(np_seed)


if __name__=="__main__":
    process_utils = ProcessData()
    process_utils.read_vocab("C:\\Users\\11415\\Desktop\\Google_Deep_Learning\\Google_NLP_DL\\9.11\\tor\\vocab\\word_vocab.txt")
    test_texts,test_words,test_labels = process_utils.read_data( \
        "C:\\Users\\11415\\Desktop\\Google_Deep_Learning\\Google_NLP_DL\\9.11\\tor\\data_plain\\test.txt")
    test_words_ids, test_labels_ids=  process_utils.convert_to_vocab(test_words,test_labels)

    texts,words,labels = process_utils.read_data( \
        "C:\\Users\\11415\\Desktop\\Google_Deep_Learning\\Google_NLP_DL\\9.11\\tor\\data_plain\\train.txt")
    words_ids, labels_ids = process_utils.convert_to_vocab(words,labels)
    datas = [[texts[i],words[i],words_ids[i],labels_ids[i]] for i in range(len(texts))]
    test_datas = [[test_texts[i],test_words[i],test_words_ids[i],test_labels_ids[i]] for i in range(len(test_texts))]

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
    word_embedding_matrix = process_utils.loadEmbMatrix(\
        "C:\\Users\\11415\\Desktop\\Google_Deep_Learning\\Google_NLP_DL\\9.11\\tor\\data_plain\\aets_embedding.txt", embedding_size, bina=False)
    print("embedding..")
    model = BiLSTM_CRF( tag_to_ix = labels_dict, \
         embedding_dim = embedding_size, hidden_dim = lstm_hidding_dim, word_embedding_matrix = word_embedding_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_func = nn.NLLLoss()
    train_epoch_loss = 0.0
    train_batches = process_utils.batch_iter(train_datas, train_labels, batch_size, embedding_size)
    each_epoch_batch_number = int(len(train_datas) / batch_size)
    train_epoch_reals = []
    train_epoch_results = []
    train_epoch_words = []
    evalute_utils = Evaluate()
    dev_max_f1 = 0.0
    for k, train_batch in enumerate(train_batches):
        model.train()
        train_texts_batch, train_words_batch, train_words_ids_batch, train_aspect_labels_batch = zip(*train_batch)
        train_aspect_labels_batch = np.array(train_aspect_labels_batch)
        x_train = torch.from_numpy(np.array(train_words_ids_batch))  ## b,83
        # debug
        x_train = x_train.long()
        out, predict_sample = model(x_train)#b,83,13,
        out = out.view(-1,classfy_number)
        train_aspect_labels_batch_reshape = np.reshape(train_aspect_labels_batch,[-1])
        batch_loss = loss_func(out, torch.from_numpy(train_aspect_labels_batch_reshape).long())
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_epoch_loss += batch_loss.item()

        train_epoch_reals.extend(train_aspect_labels_batch)
        train_epoch_results.extend(predict_sample.numpy().tolist())
        train_epoch_words.extend(train_words_batch)

        if (k + 1) % each_epoch_batch_number == 0 and k != 0:
            model.adjust_learning_rate(learning_rate, optimizer, int((k + 1) / each_epoch_batch_number ))
            print(int((k + 1) / each_epoch_batch_number )," epoch , train_loss:", train_epoch_loss)
            train_epoch_loss = 0.0
            train_precison, train_recall, train_f1 = evalute_utils.calculate(train_epoch_words, train_epoch_results,train_epoch_reals,
                                                                             )
            train_epoch_reals = []
            train_epoch_results = []
            train_epoch_words = []
            model.eval()
            dev_batches = process_utils.batch_iter(dev_datas, dev_labels, 500, 1, False)
            dev_epoch_reals,dev_epoch_predict,dev_epoch_words = [],[],[]
            for m, dev_batch in enumerate(dev_batches):
                dev_texts_batch, dev_words_batch, dev_words_ids_batch, dev_aspect_labels_batch = zip(*dev_batch)
                dev_aspect_labels_batch = np.array(dev_aspect_labels_batch)
                x_dev = torch.from_numpy(np.array(dev_words_ids_batch))
                x_dev = x_dev.long()
                out, predict_sample = model(x_dev)
                dev_epoch_reals.extend(dev_aspect_labels_batch)
                dev_epoch_predict.extend(predict_sample.numpy().tolist())
                dev_epoch_words.extend(dev_words_batch)
            dev_precison, dev_recall, dev_f1 = evalute_utils.calculate(dev_epoch_words, dev_epoch_predict,dev_epoch_reals,
                                                                            )

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
                    x_test = torch.from_numpy(np.array(test_words_ids_batch))
                    x_test = x_test.long()
                    out, predict_sample = model(x_test)
                    test_epoch_reals.extend(test_aspect_labels_batch)
                    test_epoch_predict.extend(predict_sample.numpy().tolist())
                    test_epoch_words.extend(test_words_batch)
                test_precison, test_recall, test_f1 = evalute_utils.calculate(test_epoch_words,test_epoch_predict, test_epoch_reals,
                                                                           )
                print("test evaluating...",test_precison, test_recall, test_f1)