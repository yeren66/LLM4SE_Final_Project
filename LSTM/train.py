import numpy as np
from transformers import AutoTokenizer, AdamW
from torch.utils.data import DataLoader
import torch
import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

from dataloader import MyDataset
from model import Model
# from test import test
from configs import Config
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score


def tokenizer_head(text, tokenizer, max_legnth):
    encoding = tokenizer(text, padding=True)
    if len(encoding['input_ids']) > max_legnth:
        encoding['input_ids'] = encoding['input_ids'][:max_legnth - 1] + encoding['input_ids'][-1:]
        encoding['attention_mask'] = encoding['attention_mask'][:max_legnth - 1] + encoding['attention_mask'][-1:]
    else:
        encoding['input_ids'] = encoding['input_ids'] + [0 for i in range(len(encoding['input_ids']), max_length)]
        encoding['attention_mask'] = encoding['attention_mask'] + [0 for i in range(len(encoding['attention_mask']), max_length)]
        
    return encoding


def save(model, optimizer, PATH, index):
    # 先删除文件夹，再新建文件夹，可以起到清空的作用
    if os.path.exists(PATH):
        shutil.rmtree(PATH)
    os.makedirs(PATH)
    # 保存模型参数
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(PATH, 'checkpoint' + str(index)))
    print("保存模型参数")


def load(model, PATH):
    checkpoint = torch.load(PATH)
    model.module.load_state_dict(checkpoint['model_state_dict'], False)
    print("加载further pretrained模型成功")
    return model

result_label = []
result_index = []
result_pred = []
result_labels = []
result_indexs = []
result_preds = []
tmp_index = []
tmp_pred = []
tmp_label = []
accs = [0,0,0,0,0]
prcs = [0,0,0,0,0]
rcs =[0,0,0,0,0]
f1s = [0,0,0,0,0]
aucs = [0,0,0,0,0]

def evl(model, eval_loader):
    tmp_index.clear()
    tmp_pred.clear()
    tmp_label.clear()
    y_true = []
    y_score = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            out = torch.sigmoid(model(input_ids, attention_mask))

            y_true.append(labels.item())
            y_score.append(out.item())
            
            eval_index = batch['index'].numpy()[0]
            tmp_index.append(eval_index)
            eval_label = batch['labels'].numpy()[0]
            tmp_label.append(eval_label)
            if out.item() >= 0.5:
                tmp_pred.append(1)
            else:
                tmp_pred.append(0)
            

    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=1)
    auc_ = auc(fpr, tpr)
    y_pred = [1 if p >= 0.5 else 0 for p in y_score]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    return acc, prc, rc, f1, auc_

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
    labels = [1 if label == 1 else 0 for label in labels]
    
    # Create an index list for the entries
    index = [idx + 1 for idx in range(len(labels))]
    
    return reviews, labels, index

from tqdm import tqdm
import numpy as np

def train(model, train_loader, eval_loader, optim, loss_function, max_epoch, start_epoch, patience):
    max_acc = 0
    no_improve_epoch = 0
    print('-------------- start training ---------------', '\n')
    for epoch in tqdm(range(max_epoch), desc="Epochs", leave=True):
        # Skip epochs before start_epoch
        if epoch < start_epoch:
            continue
        tqdm.write(f"========= Epoch: {epoch} ==============")
        
        epoch_losses = []
        for batch in tqdm(train_loader, desc="Training", leave=False):
            # Zero the gradients
            optim.zero_grad()

            # Move data to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            out = model(input_ids, attention_mask)
            loss = loss_function(out, labels.float())

            # Store loss
            epoch_losses.append(loss.item())
            # tqdm.write(f"[{len(epoch_losses)}/{len(train_loader)}] Loss: {loss.item():.3f}")

            # Backward pass and optimize
            loss.backward()
            optim.step()

        # Average loss for the epoch
        avg_loss = np.mean(epoch_losses)
        tqdm.write(f"Average Loss: {avg_loss:.3f}")

        # Validation logic
        if epoch % 1 == 0:
            model.eval()
            acc, prc, rc, f1, auc = evl(model=model, eval_loader=eval_loader)
            model.train()
            if max_acc < acc:
                # Assuming these are defined elsewhere in your script
                max_acc = acc
                save(model, optim, config.model_save_path, epoch)
                tqdm.write(f'New max accuracy: {max_acc:.3f}')
            else:
                no_improve_epoch += 1
                tqdm.write(f"No improvement: {no_improve_epoch} epochs")

            # Early stopping
            if no_improve_epoch >= patience:
                tqdm.write("Stopping early due to no improvement")
                break
                
    tqdm.write('++++++++++++++++++ Training Complete ++++++++++++++++++')


if __name__ == '__main__':
    # 配置类
    config = Config()
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    # 模型最长输入
    max_length = config.max_length
    
    # 加载数据
    
    data_train_path = config.data_train_path
    data_eval_path = config.data_eval_path
    train_reviews, train_labels, train_index = load_and_process_jsonl(data_train_path)
    eval_reviews, eval_labels, eval_index = load_and_process_jsonl(data_eval_path)
    print("训练集:", len(train_labels))
    print("测试集:", len(eval_labels))

    train_dataset = MyDataset(tokenizer_head, tokenizer, max_length, train_reviews, train_labels, train_index)
    eval_dataset = MyDataset(tokenizer_head, tokenizer, max_length, eval_reviews, eval_labels, eval_index)

    # 生成训练和测试Dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=True)

    # 模型
    model = Model(config)
    # 定义GPU/CPU
    device = config.device
    model.to(device)
    # 多GPU并行
    model = torch.nn.DataParallel(model, device_ids=config.device_ids)
    #    model = torch.nn.DataParallel(model)
    # 加载已有模型参数
    if config.start_epoch > 0:
        model = load(model, config.pretrained_model_path)
    # 训练模式
    model.train()
    # 训练次数
    max_epoch = config.num_epoch
    # 开始训练是第几轮
    start_epoch = config.start_epoch
    # 优化器
    optim = AdamW(model.parameters(), lr=5e-5)
    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss()
    
    # 开始训练
    train(model=model, train_loader=train_loader, eval_loader=eval_loader, optim=optim, loss_function=loss_function,
            max_epoch=max_epoch, start_epoch=start_epoch, patience=config.patience)







