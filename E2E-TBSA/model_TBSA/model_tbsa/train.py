from process_lap_data import ProcessData,Evaluate
from TBSA import TBSA
import torch
from config import  *
import torch.nn as nn
import numpy as np
import numpy
import random
from torch.backends import cudnn
#CUBLAS_WORKSPACE_CONFIG=:4096:8
if torch.cuda.is_available():
    device = torch.device('cuda:0')
random.seed(seed)
np.random.seed(np_seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#if deterministic:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
mpqa_vocab = {}
with open("C:\\Users\\11415\\Desktop\\github repo\\Google_NLP_DL\\E2E-TBSA\\model_TBSA\\vocab\\mpqa_full.txt",encoding='utf-8') as f:
    for line in f.readlines():
        mpqa_vocab[line.strip().split('\t')[0]] =  int(line.strip().split('\t')[1])
def get_transition_init():
    transition_scores = np.zeros((5, 13))
    for t in transition_path:
        next_tags = transition_path[t]
        n_next_tag = len(next_tags)
        ote_id = boundary_dict[t]
        for nt in next_tags:
            ts_id = labels_dict[nt]
            transition_scores[ote_id][ts_id] = 1.0 / n_next_tag

    return transition_scores

def make_one_hot(data,number):
    data = np.array(data)
    batch_one_hot  = []
    for each in data:
        batch_one_hot.append((numpy.arange(number)==each[:,None]).astype(numpy.integer))
    return np.array(batch_one_hot)

def get_oe_labels(words):
    oe_labels_batch = []
    for each_word in words:
        each_oe_label = []
        for s, lab in enumerate(each_word):
            ifopinion = False
            if s == 0:
                windows = each_word[:1]
            elif s == len(each_word) - 1:
                windows = each_word[-2:]
            else:
                windows = each_word[s-1:s+2]
            for ww in windows:
                if ww in mpqa_vocab:
                 #   print(ww)
                    ifopinion = True
                    break
            if ifopinion:
                each_oe_label.append(1)
            else:
                each_oe_label.append(0)

        oe_labels_batch.append(each_oe_label)
    return oe_labels_batch

if __name__=="__main__":
    
    process_utils = ProcessData()
    process_utils.read_vocab("C:\\Users\\11415\\Desktop\\github repo\\Google_NLP_DL\\E2E-TBSA\\model_TBSA\\vocab\\word_vocab.txt")
    test_texts,test_words,test_labels,test_boundary_labels = process_utils.read_data( \
        "C:\\Users\\11415\\Desktop\\github repo\\Google_NLP_DL\\E2E-TBSA\\model_TBSA\\data_plain\\test.txt")
    test_words_ids, test_labels_ids,test_boundary_labels_ids =  process_utils.convert_to_vocab(test_words,test_labels,test_boundary_labels)

    texts,words,labels,boundary_labels = process_utils.read_data(
        "C:\\Users\\11415\\Desktop\\github repo\\Google_NLP_DL\\E2E-TBSA\\model_TBSA\\data_plain\\train.txt")
    words_ids, labels_ids,boundary_labels_ids = process_utils.convert_to_vocab(words,labels,boundary_labels)
    datas = [[texts[i],words[i],words_ids[i],labels_ids[i],boundary_labels_ids[i]] for i in range(len(texts))]
    test_datas = [[test_texts[i],test_words[i],test_words_ids[i],test_labels_ids[i],test_boundary_labels_ids[i]] for i in range(len(test_texts))]

    val_sample_ids = np.random.choice(len(datas),  int(len(datas) * 0.1), replace=False)
    train_datas,dev_datas = [],[]
    for i,data in enumerate(datas):
        if i in val_sample_ids:
            dev_datas.append(data)
        else:
            train_datas.append(data)
    print("train_dev_splited..")
    word_embedding_matrix = process_utils.loadEmbMatrix("C:\\Users\\11415\\Desktop\\github repo\\Google_NLP_DL\\E2E-TBSA\\model_TBSA\\data_plain\\aets_embedding.txt", embedding_size, bina=False)
    print("embedding..")
    if torch.cuda.is_available():
        word_embedding_matrix = torch.from_numpy(word_embedding_matrix).to(device)
    else:
        word_embedding_matrix = torch.from_numpy(word_embedding_matrix)

    transition_scores = get_transition_init()
    if torch.cuda.is_available():
        transition_scores = torch.from_numpy(transition_scores).to(torch.float32).to(device)
        model = TBSA(len(word_embedding_matrix), embedding_size, word_embedding_matrix, transition_scores).to(device)
    else:
        transition_scores = torch.from_numpy(transition_scores).to(torch.float32)
        model = TBSA(len(word_embedding_matrix), embedding_size, word_embedding_matrix, transition_scores)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_func = nn.NLLLoss()
    train_epoch_loss = 0.0
    train_batches = process_utils.batch_iter(train_datas, batch_size, epoch)
    each_epoch_batch_number = int(len(train_datas) / batch_size)
    evalute_utils = Evaluate()
    dev_max_f1 = 0.0
    for k, train_batch in enumerate(train_batches):
        model.train()
        train_texts_batch, train_words_batch, train_words_ids_batch, train_aspect_labels_batch,train_boundary_labels_batch = zip(*train_batch)
        if torch.cuda.is_available():
            x_train = torch.from_numpy(np.array(train_words_ids_batch)).long().to(device)  ## b,83
            train_oe_labels_batch = torch.from_numpy(make_one_hot(get_oe_labels(train_words_batch), 2)).long().to(device)
            predict_sample, batch_loss = model(x_train,torch.from_numpy(make_one_hot(train_aspect_labels_batch, number=13)).long().to(device),
                                               torch.from_numpy(make_one_hot(train_boundary_labels_batch, number=5)).long().to(device), train_oe_labels_batch)  # b,83,13,

        else:
            x_train = torch.from_numpy(np.array(train_words_ids_batch))## b,83
            train_oe_labels_batch = torch.from_numpy(make_one_hot(get_oe_labels(train_words_batch), 2))
            predict_sample, batch_loss = model(x_train,torch.from_numpy(make_one_hot(train_aspect_labels_batch, number=13)),torch.from_numpy(make_one_hot(train_boundary_labels_batch, number=5)), train_oe_labels_batch)  # b,83,13,

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_epoch_loss += batch_loss.item()

        if (k + 1) % each_epoch_batch_number == 0 and k != 0:
            model.adjust_learning_rate(learning_rate, optimizer, int((k + 1) / each_epoch_batch_number ))
            print(int((k + 1) / each_epoch_batch_number )," epoch , train_loss:", train_epoch_loss)
            train_epoch_loss = 0.0
            model.eval()
            dev_batches = process_utils.batch_iter(dev_datas, 500, 1, False)
            dev_epoch_reals,dev_epoch_predict,dev_epoch_words = [],[],[]
            for m, dev_batch in enumerate(dev_batches):
                dev_texts_batch, dev_words_batch, dev_words_ids_batch, dev_aspect_labels_batch,dev_boundary_labels_batch = zip(*dev_batch)
                if torch.cuda.is_available():
                    dev_oe_labels_batch = torch.from_numpy(make_one_hot(get_oe_labels(dev_words_batch),2)).to(device)
                    x_dev = torch.from_numpy(np.array(dev_words_ids_batch)).to(device)
                    predict_sample, _ = model(x_dev,torch.from_numpy(make_one_hot(dev_aspect_labels_batch, number=13)).to(device),torch.from_numpy(make_one_hot(dev_boundary_labels_batch, number=5)).to(device), dev_oe_labels_batch)
                else:
                    dev_oe_labels_batch = torch.from_numpy(make_one_hot(get_oe_labels(dev_words_batch), 2))
                    x_dev = torch.from_numpy(np.array(dev_words_ids_batch))
                    predict_sample, _ = model(x_dev,torch.from_numpy(make_one_hot(dev_aspect_labels_batch, number=13)),torch.from_numpy(make_one_hot(dev_boundary_labels_batch, number=5)), dev_oe_labels_batch)

                dev_epoch_reals.extend(dev_aspect_labels_batch)
                dev_epoch_predict.extend(predict_sample.cpu().numpy().tolist())
                dev_epoch_words.extend(dev_words_batch)
            dev_precison, dev_recall, dev_f1 = evalute_utils.calculate(dev_epoch_words, dev_epoch_predict,dev_epoch_reals,
                                                                            )

            if dev_f1>dev_max_f1:
                dev_max_f1 = dev_f1
                print("dev evaluating...", dev_precison, dev_recall, dev_f1)
                #######test
                model.eval()
                test_batches = process_utils.batch_iter(test_datas, 500, 1, False)
                test_epoch_reals, test_epoch_predict, test_epoch_words = [], [], []
                for p, test_batch in enumerate(test_batches):
                    test_texts_batch, test_words_batch, test_words_ids_batch, test_aspect_labels_batch,test_boundary_labels_batch = zip(*test_batch)
                    if torch.cuda.is_available():
                        test_oe_labels_batch = torch.from_numpy(make_one_hot(get_oe_labels(test_words_batch),2)).to(device)
                        x_test = torch.from_numpy(np.array(test_words_ids_batch)).to(device)
                        predict_sample, _ = model(x_test, torch.from_numpy(make_one_hot(test_aspect_labels_batch, number=13)).to(device), torch.from_numpy(make_one_hot(test_boundary_labels_batch, number=5)).to(device), test_oe_labels_batch)
                    else:
                        test_oe_labels_batch = torch.from_numpy(make_one_hot(get_oe_labels(test_words_batch), 2))
                        x_test = torch.from_numpy(np.array(test_words_ids_batch))
                        predict_sample, _ = model(x_test, torch.from_numpy(make_one_hot(test_aspect_labels_batch, number=13)), torch.from_numpy(make_one_hot(test_boundary_labels_batch, number=5)), test_oe_labels_batch)

                    test_epoch_reals.extend(test_aspect_labels_batch)
                    test_epoch_predict.extend(predict_sample.cpu().numpy().tolist())
                    test_epoch_words.extend(test_words_batch)
                test_precison, test_recall, test_f1 = evalute_utils.calculate(test_epoch_words,test_epoch_predict, test_epoch_reals,ifwriter=True
                                                                           )
                print("test evaluating...",test_precison, test_recall, test_f1)
