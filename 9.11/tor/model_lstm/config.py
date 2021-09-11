batch_size = 16
learning_rate = 0.001
decay_rate = 0.05
seed = 1314159
np_seed = 7894
embedding_size = 300
classfy_number = 13
dropout = 0.5
lstm_hidding_dim = 50

labels_dict = {
    'O':0,'B-POS':1,'I-POS':2,'E-POS':3,'S-POS':4,'B-NEG':5,'I-NEG':6,'E-NEG':7,'S-NEG':8,'B-NEU':9,
    'I-NEU':10,'E-NEU':11,'S-NEU':12}
sentiments_dict = {1:'POS',2:'POS',3:'POS',4:'POS',5:'NEG',6:'NEG',7:'NEG',8:'NEG',9:'NEU',10:'NEU',11:'NEU',12:'NEU'}
