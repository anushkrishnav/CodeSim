from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset


class CodePairsDataset(Dataset):
    def __init__(self, codes1, codes2, labels):
        self.codes1 = codes1
        self.codes2 = codes2
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.df = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.codes1[idx], self.codes2[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        return {**{k: v.squeeze(0) for k, v in encoding.items()}, 'labels': torch.tensor(self.labels[idx])}
    
    def save_tokenizer(self, path):
        self.tokenizer.save_pretrained(path)

    @staticmethod
    def get_loader(df, x_col, y_col):
        X = df[x_col]
        y = df[y_col]
        dataset = CodePairsDataset(X['code1'].values, X['code2'].values, y.values)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        dataset.save_tokenizer('models/Clonebert/tokenizer-code-bert')
        return loader



