import numpy as np
from transformers import AutoTokenizer, AdamW
from torch.utils.data import DataLoader
import torch
import os
import shutil
from pathlib import Path
import json

from dataloader import MyDataset
from model import Model
from configs import Config
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score



def tokenizer_head(text, tokenizer, max_legnth):
    encoding = tokenizer(text, padding=True)
    if len(encoding['input_ids']) > max_legnth:
        encoding['input_ids'] = encoding['input_ids'][:max_legnth - 1] + encoding['input_ids'][-1:]
        #        encoding['token_type_ids'] = encoding['token_type_ids'][:max_legnth-1] + encoding['token_type_ids'][-1:]
        encoding['attention_mask'] = encoding['attention_mask'][:max_legnth - 1] + encoding['attention_mask'][-1:]
        # encoding.pop('token_type_ids')
    else:
        encoding['input_ids'] = encoding['input_ids'] + [0 for i in range(len(encoding['input_ids']), max_length)]
        encoding['attention_mask'] = encoding['attention_mask'] + [0 for i in
                                                                   range(len(encoding['attention_mask']), max_length)]
        # encoding.pop('token_type_ids')
    return encoding


def load(model, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def test(model, test_loader,device):
    y_true = []
    y_score = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            out = torch.sigmoid(model(input_ids, attention_mask))

            y_true.append(labels.item())
            y_score.append(out.item())

    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=1)
    auc_ = auc(fpr, tpr)
    y_pred = [1 if p >= 0.5 else 0 for p in y_score]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))


def load_and_process_jsonl(file_path):
    # Initialize lists to store the reviews and labels
    reviews = []
    labels = []
    
    # Open the JSONL file and read line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            reviews.append(data['review'])
            labels.append(data['label'])
    
    # Convert reviews to lower case
    reviews = [review.lower() for review in reviews]
    
    # Correct label if needed (reverse 1 to 0 and vice versa)
    labels = [0 if label == 1 else 1 for label in labels]
    
    # Create an index list for the entries
    index = [idx + 1 for idx in range(len(labels))]
    
    return reviews, labels, index

if __name__ == '__main__':
    # 配置类
    config = Config()
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    # 模型最长输入
    max_length = config.max_length

    data_test_path = config.data_test_path
    test_reviews, test_labels, test_index = load_and_process_jsonl(data_test_path)

    test_dataset = MyDataset(tokenizer_head, tokenizer, max_length, test_reviews, test_labels, test_index)

    # 生成训练和测试Dataloader
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True)

    # 模型
    model = Model(config)
    model = load(model, config.model_test_path)
    # 定义GPU/CPU
    device = config.device
    model.to(device)
    # 多GPU并行
    model = torch.nn.DataParallel(model, device_ids=config.device_ids)

    # 测试
    test(model, test_loader,device=device)








