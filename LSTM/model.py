import numpy as np
from torch import nn, Tensor
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import *

class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.bert = AutoModel.from_pretrained(config.model_path)

        # lstm 模型
        self.lstm = nn.LSTM(input_size=config.input_size, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True,
                            dropout=config.dropout, bias=True, bidirectional=True)
        self.lstm.flatten_parameters()
        # dropout
        self.dropout = nn.Dropout(config.dropout)
        # linear 全连接层：双向LSTM要*2
        self.fc = nn.Linear(config.hidden_size * 2,
                            config.num_classes)  # 自定义全连接层 ，输入数（输入的最后一个维度），输出数（多分类数量），bert模型输出的最后一个维度是1024，这里的输入要和bert最后的输出统一

    def forward(self, input_ids, attention_mask):

        output = self.bert(input_ids, attention_mask=attention_mask)[1]
        # print(input_ids.type())
        # output = input_ids.float()
        # 将词向量输入lstm
        out, _ = self.lstm(output)

        # 将lstm输入进行dropout，其输入输出shape相同
        out = self.dropout(out)

        # 全连接
        # print(out.shape)
        # out = out[:, :]  # 只要序列中最后一个token对应的输出，（因为lstm会记录前边token的信息）
        # print(out.shape)
        out = self.fc(out).squeeze(-1)

        return out



