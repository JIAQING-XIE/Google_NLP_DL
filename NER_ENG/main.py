from data_process import Data, Glove
import argparse
import torch
import torch.utils.data as DATA
from evaluate import BILSTM_Model, bilstm_train_and_eval
torch.manual_seed(131415)

def data_batch(x, y, batch_size):
    dataset = DATA.TensorDataset(x, y)
    loader = DATA.DataLoader(dataset, batch_size, shuffle= True, num_workers= 4)
    return loader

if __name__ == "__main__":
    # add arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--decay_rate', type=int, default=0.05)   
    parser.add_argument('--lstm_dim', type=int, default=50)  
    parser.add_argument('--drop_out', type=int, default=0.5)  
    args=parser.parse_args()
    
    train_data = Data("train.txt")
    test_data = Data("test.txt")
    
    train_word_lists, train_tag_lists = train_data.transform()
    test_word_lists, test_tag_lists = train_data.transform()

    train_tag_lists = train_data.statistics(train_tag_lists)
    train_word_lists, valid_word_lists, train_tag_lists, valid_tag_lists = train_data.train_valid_split(
        train_word_lists, train_tag_lists)
    train_word_lists, train_tag_lists, train_word2id, train_tag2id = train_data.to_id(train_word_lists, train_tag_lists, make_vocab=True)
        # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    bilstm_train_word2id, bilstm_train_tag2id = train_data.extend_maps(train_word2id, train_tag2id, for_crf=False)

    lstm_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (valid_word_lists, valid_tag_lists),
        (test_word_lists, test_tag_lists),
        bilstm_train_word2id,  bilstm_train_tag2id,
        crf=False, lr=args.learning_rate, batch_size=args.batch_size
    )
    
    #train_loader = data_batch(X_train, y_train, args.batch_size)
    #for step, (x,y) in train_loader:
    #    print(step)
    
    #data = Glove()
    #model = data.okk("C:\\Users\\11415\\Desktop\\Google_Deep_Learning\\Google_NLP_DL\\NER_ENG\\glove.840B.300d.word2vec.txt")
    