import json
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np

# 初始化 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('/mnt/sdb/zhaoyuan/zhangquanjun/sy/LLM4SE_Final_Project/LSTM/bert')

def get_review_lengths(jsonl_file):
    lengths = []
    with open(jsonl_file, 'r') as f:
        for line in tqdm(f, desc="Processing reviews"):
            data = json.loads(line)
            review = data.get('review', '')
            # 使用 BERT tokenizer 编码 review
            encoded_review = tokenizer.encode(review, add_special_tokens=True)
            lengths.append(len(encoded_review))
    return lengths

def plot_length_distribution(lengths, output_file):
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Review Lengths After BERT Tokenization')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.savefig(output_file)  # 保存图像到文件
    plt.close()

# 指定 JSONL 文件路径
jsonl_file = '/mnt/sdb/zhaoyuan/zhangquanjun/sy/LLM4SE_Final_Project/LSTM/data/IMDB.jsonl'
output_file = 'length_distribution.png'

# 获取 review 的长度分布
review_lengths = get_review_lengths(jsonl_file)

# 计算平均长度
average_length = np.mean(review_lengths)
print(f'Average length of encoded reviews: {average_length:.2f}')

# 计算小于512长度的token占所有长度token的比例
less_than_512 = sum(1 for length in review_lengths if length < 512)
total_reviews = len(review_lengths)
proportion_less_than_512 = less_than_512 / total_reviews * 100
print(f'Proportion of reviews with length less than 512 tokens: {proportion_less_than_512:.2f}%')

# 绘制并保存长度分布图
plot_length_distribution(review_lengths, output_file)
print(f'Distribution plot saved as {output_file}')