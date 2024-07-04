from transformers import BertModel, AdamW, BertTokenizer
import torch

class Config(object):
    '''
    配置参数
    '''
    def __init__(self):
        # 数据路径
        self.data_train_path = 'data/train.jsonl'
        self.data_eval_path = 'data/eval.jsonl'
        self.data_test_path = 'data/test.jsonl'
        # 模型保存路径
        self.model_save_path = './output/'

        # 模型测试路径
        self.model_test_path = '/mnt/sdb/zhaoyuan/zhangquanjun/sy/LLM4SE_Final_Project/LSTM/output/checkpoint1'

        # GPU 配置使用检测
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_ids = [0]
        # GPU 是否使用cuda
        self.use_cuda = True

        # bert 预训练模型
        self.model_path = '/mnt/sdb/zhaoyuan/zhangquanjun/sy/LLM4SE_Final_Project/LSTM/bert'

        # 模型的最长输入，bert为512，longformer为4096
#        self.max_length = 4096
        self.max_length = 512

        self.patience = 2

        # lstm 输入数据特征维度：Bert模型 token的embedding维度 = Bert模型后接自定义分类器（单隐层全连接网络）的输入维度
        self.input_size = 768
        # lstm 隐层维度
        self.hidden_size = 256
        # lstm 循环神经网络层数
        self.num_layers = 2
        # dropout：按一定概率随机将神经网络单元暂时丢弃，可以有效防止过拟合
        self.dropout = 0.5

        # linear 输入特征size
        self.num_classes = 1

        # epoch 整体训练次数
        self.num_epoch = 50
        # epoch 开始训练时已处于第几次，默认为0
        self.start_epoch = 0
        # batch 训练batch大小
        self.train_batch_size = 32
        # batch 测试batch大小
        self.eval_batch_size = 1
        # batch 测试batch大小
        self.test_batch_size = 1

