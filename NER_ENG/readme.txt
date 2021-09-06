0、训练测试数据需要进行预处理，主要问题存在于“####”左边的文本分词后和右边的标注词的长度不一致，即二者存在误差。
   这里需要自己处理一下，以及分词需要自己用re写一下，保证和右边的标注结果一致。

1、论文没给训练和验证具体划分文件，随机取10%，最终dev train test的分布如下（自己尝试尽量拟合）：
    train dev test
POS  883  104  339
NEG  754  106  130
NEU  404  46   165

我复现的随机种子为30244供参考。

2、评价指标为micro-f1，recall和precison要针对实体级别来计算，比如：
正确标签: B-POS I-POS E-POS O O S-POS 0 B-NEG E-NEG 0 S-NEU
预测标签: B-POS I-POS I-POS O O S-POS B-POS E-POS O O S-POS
正确的实体有4个，预测出的有3个，正确的仅有1个，所以recall = 1/4 precison = 1/3  f1 = 2*precison*recall/(p+r)

3、固定所有的随机参数，即固定网络，方便复现，以下随机参数供参考：
numpy_seed = 13456
tf_seed = 131415
batch_size = 8

4、网络参数参考：
learning_rate = 0.001
decay_rate = 0.05
lstm_dim = 50
drop_out = 0.5
学习率衰退的策略采用反时限衰减（倒数衰减的方式），词向量层加一层dropout。

5、采用glove.300作为预训练词向量，oov的词采用uniform(-0.25,0.25)，网络采用的是双向lstm。

6、还有一些小的trick和调参，自己尝试一下。

7、最终结果采用在验证集合取得最高f1的那一轮网络，然后再测试集上测试，结果接近于precison = 56.30 recall = 47.95 f1 = 51.79

（论文中为57.91,46.21,51.40），结果差不多即可。

加上crf后f1在54左右
