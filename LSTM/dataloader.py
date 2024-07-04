from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, func, tokenizer, max_length, texts, labels, index):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.func = func
        self.texts = texts
        self.labels = labels
        self.index = index

    def _encode(self, text):
        return self.func(text, self.tokenizer, self.max_length)

    def __getitem__(self, idx):
        index = self.index[idx]
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self._encode(text)

        item = dict()
        for key, val in encoding.items():
            if key.startswith("token_type_ids"):
                continue
            item[key] = torch.tensor(val)
        item['labels'] = torch.tensor(label)
        item['index'] = torch.tensor(index)
        return item

    def __len__(self):
        return len(self.labels)
