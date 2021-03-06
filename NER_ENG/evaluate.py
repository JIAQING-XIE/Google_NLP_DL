import torch
from model import BiLSTM
import torch.optim as optim
from utils import cal_loss, sort_by_lengths, tensorized, save_model
from copy import deepcopy
import time
from evaluating import Metrics

def count_f1_score(pred, target):
    # 需要计算的东西： precision, recall
    # precision : 预测正确/ 总的预测实体个数
    correct = 0 
    predict_total = 0
    target_total = 0
    target_aspects = []
    predict_aspects = []
    for i in range(len(pred)):
        predict_aspect = []
        for j in range(len(pred[i])):
            if pred[i][j] == 'O': # 不是实体，跳过
                continue
            elif pred[i][j][0] == 'S': #单个实体
                predict_aspect.append((j, pred[i][j][2:]))
                predict_total+=1
            elif pred[i][j][0] == 'B': #多个实体
                nst_E = j 
                for nearest_E in range(j+1, len(pred[i])):
                    if pred[i][nearest_E][0] == 'E':
                        nst_E = nearest_E
                        break
                sentiment = []
                for cc in range(j, nst_E+1):
                    sentiment.append(pred[i][cc][2:])
                sentiment = tuple(sentiment)
                predict_aspect.append((j, nst_E, sentiment))
                predict_total+=1
        predict_aspects.append(predict_aspect)

    for i in range(len(target)):
        target_aspect = []
        for j in range(len(target[i])):
            if target[i][j] == 'O': # 不是实体，跳过
                continue
            elif target[i][j][0] == 'S': #单个实体
                target_aspect.append((j, target[i][j][2:]))
                target_total+=1
            elif target[i][j][0] == 'B': #多个实体
                nst_E = j 
                for nearest_E in range(j+1, len(target[i])):
                    if target[i][nearest_E][0] == 'E':
                        nst_E = nearest_E
                        break
                sentiment = []
                for cc in range(j,nst_E+1):
                    sentiment.append(target[i][cc][2:])
                sentiment = tuple(sentiment)
                target_aspect.append((j, nst_E, sentiment))
                target_total+=1
        target_aspects.append(target_aspect)         

    for sen_idx in range(len(predict_aspects)):
        for a_idx in range(len(predict_aspects[sen_idx])):
            if predict_aspects[sen_idx][a_idx] in target_aspects[sen_idx]:
                correct+=1

    precision = 0
    if predict_total != 0:        
        precision = correct / predict_total
    else:
        precision = 0
    recall = 0
    if target_total != 0:
        recall = correct / target_total
    else:
        recall = 0
    f1 = 0
    if precision + recall != 0.0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    
    return precision, recall, f1

class BILSTM_Model(object):
    def __init__(self, weight, vocab_size, out_size, lr, batch_size, crf=True):
        """功能：对LSTM的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类
            crf选择是否添加CRF层"""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型参数
        self.emb_size = 300
        self.hidden_size = 128

        self.crf = crf
        # 根据是否添加crf初始化不同的模型 选择不一样的损失计算函数
        if not crf:
            self.model = BiLSTM(vocab_size, self.emb_size,
                                self.hidden_size, out_size, weight).to(self.device)
            self.cal_loss_func = cal_loss
        #else:
        #    self.model = BiLSTM_CRF(vocab_size, self.emb_size,
        #                            self.hidden_size, out_size).to(self.device)
        #    self.cal_loss_func = cal_lstm_crf_loss

        # 加载训练参数：
        self.epoches = 100
        self.print_step = 5
        self.lr = lr
        self.batch_size = batch_size

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self._model = None
        self.best_model = None

    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists,
              word2id, tag2id):
        # 对数据集按照长度进行排序
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(
            dev_word_lists, dev_tag_lists)

        B = self.batch_size
        best_valid_epoch = 0
        best_valid_f1 = 0
        no_better_f1_rounds = 0
        for e in range(1, self.epoches+1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind:ind+B]
                batch_tags = tag_lists[ind:ind+B]

                losses += self.train_step(batch_sents,
                                          batch_tags, word2id, tag2id)

                if self.step % self.print_step == 0:
                    total_step = (len(word_lists) // B + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.
            # 每轮结束测试在验证集上的性能，保存最好的一个
            self._model = deepcopy(self.model)
            val_loss = self.validate(
                dev_word_lists, dev_tag_lists, word2id, tag2id)
            pred_val_tags_lists = self.test(dev_word_lists, dev_tag_lists, word2id, tag2id) # 模型在vad集上的prediction tag list
            val_precision, val_recall, val_f1 = self.count_f1_score(pred_val_tags_lists, dev_tag_lists)
            if val_f1 > best_valid_f1:
                best_valid_f1 = val_f1
                best_valid_epoch = e
                self.best_model = self._model
                self._best_val_loss = val_loss
                no_better_f1_rounds = 0
            else:
                if no_better_f1_rounds == 100:
                    break
                else:
                    no_better_f1_rounds+=1

        print("Best Epoch {}, Val Loss:{:.4f}, F1 Score:{:.4f}".format(best_valid_epoch \
                , val_loss, best_valid_f1))

    def count_f1_score(self, pred, target):
        # 需要计算的东西： precision, recall
        # precision : 预测正确/ 总的预测实体个数
        correct = 0 
        predict_total = 0
        target_total = 0
        target_aspects = []
        predict_aspects = []
        for i in range(len(pred)):
            predict_aspect = []
            for j in range(len(pred[i])):
                if pred[i][j] == 'O': # 不是实体，跳过
                    continue
                elif pred[i][j][0] == 'S': #单个实体
                    predict_aspect.append((j, pred[i][j][2:]))
                    predict_total+=1
                elif pred[i][j][0] == 'B': #多个实体
                    nst_E = j 
                    for nearest_E in range(j+1, len(pred[i])):
                        if pred[i][nearest_E][0] == 'E':
                            nst_E = nearest_E
                            break
                    sentiment = []
                    for cc in range(j, nst_E+1):
                        sentiment.append(pred[i][cc][2:])
                    sentiment = tuple(sentiment)
                    predict_aspect.append((j, nst_E, sentiment))
                    predict_total+=1
            predict_aspects.append(predict_aspect)

        for i in range(len(target)):
            target_aspect = []
            for j in range(len(target[i])):
                if target[i][j] == 'O': # 不是实体，跳过
                    continue
                elif target[i][j][0] == 'S': #单个实体
                    target_aspect.append((j, target[i][j][2:]))
                    target_total+=1
                elif target[i][j][0] == 'B': #多个实体
                    nst_E = j 
                    for nearest_E in range(j+1, len(target[i])):
                        if target[i][nearest_E][0] == 'E':
                            nst_E = nearest_E
                            break
                    sentiment = []
                    for cc in range(j, nst_E+1):
                        sentiment.append(target[i][cc][2:])
                    sentiment = tuple(sentiment)
                    target_aspect.append((j, nst_E, sentiment))
                    target_total+=1
            target_aspects.append(target_aspect)         

        for sen_idx in range(len(predict_aspects)):
            for a_idx in range(len(predict_aspects[sen_idx])):
                print(predict_aspects[sen_idx])
                print(target_aspects[sen_idx])
                if predict_aspects[sen_idx][a_idx] in target_aspects[sen_idx]:
                    correct+=1

        precision = 0
        if predict_total != 0:        
            precision = correct / predict_total
        else:
            precision = 0
        recall = 0
        if target_total != 0:
            recall = correct / target_total
        else:
            recall = 0
        f1 = 0
        if precision + recall != 0.0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        return precision, recall, f1

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1
        # 准备数据
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, lengths = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)

        # forward
        scores = self.model(tensorized_sents, lengths)

        # 计算损失 更新参数
        self.optimizer.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind+self.batch_size]
                batch_tags = dev_tag_lists[ind:ind+self.batch_size]
                tensorized_sents, lengths = tensorized(
                    batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)

                # forward
                scores = self.model(tensorized_sents, lengths)

                # 计算损失
                loss = self.cal_loss_func(
                    scores, targets, tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """返回最佳模型在测试集上的预测结果"""
        # 准备数据
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self._model.eval()
        with torch.no_grad():
            batch_tagids = self._model.test(
                tensorized_sents, lengths, tag2id)

        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.crf:
                for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists
    
def bilstm_train_and_eval(weight, train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=True, lr = 0.001, batch_size = 8):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(weight, vocab_size, out_size, lr, batch_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "C:\\Users\\11415\\Desktop\\Google_Deep_Learning\\Google_NLP_DL\\NER_ENG\\ckpts\\"+model_name+".pkl")

    print("End of Training....Total time {} seconds.".format(int(time.time()-start)))
    
    print("Evaluating {} Model...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()
    
    return pred_tag_lists