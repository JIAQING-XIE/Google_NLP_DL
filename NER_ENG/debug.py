def debug_1(group1, group2):
    new_group = []
    i = 0
    j = 0
    while i < len(group1):
        while j < len(group2):
            if group1[i] == group2[j]:
                new_group.append(group1[i])
                i+=1 # move to next
                j+=1 # move to next
            elif group1[i] in group2[j]: # meet mark behind
                current = i
                for current in range(i, len(group1) + 1):
                    if ''.join(group1[i:current]) in group2[j]:
                        if ''.join(group1[i:current]) == group2[j]:
                            new_group.append(group2[j])
                            i = current
                            j+=1
                            break
                    else:
                        break  
    return new_group

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

if __name__ == "__main__":
    """
    group1 = ["1", "-", "1", "-", "1", "group", "has", "a", "lot", "-", "common", "-", "dog"]
    group2 = ["1-1-1", "group", "has", "a", "lot-common-", "dog"]
    ans = debug_1(group1, group2)
    print("group1:{}".format(group1))
    print("group2:{}".format(group2))
    print(ans)
    """
    pred = [['O', 'B-NEG', 'E-NEG', 'O', 'O', 'B-NEG', 'I-NEG', 'O', 'O', 'S-POS']]
    target = [['O', 'B-NEG', 'E-NEG', 'S-NEG', 'O', 'B-NEG', 'I-NEG', 'E-NEG', 'O', 'S-POS']]
    p,r,f1 = count_f1_score(pred, target)
    print(f1)