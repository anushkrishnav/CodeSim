import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

df = pd.read_parquet('data/train/clone-detection-600k-5fold.parquet')
# first 2000 rows
df = df.head(100)
columns = ["code1", "code2", "similar"]
df = df[columns]
# drop columns with NaN
df = df.dropna()
# train test split, target column being the similar column
X = df.drop(columns=['similar'])
y = df['similar']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fine tune code-bert using the train set to act as an encoder
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset

import torch

# tokenize the code 
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
def tokenize_pair(code_1, code_2):
    return tokenizer(code_1, code_2, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

class CodePairsDataset(Dataset):
    def __init__(self, codes1, codes2, labels):
        self.codes1 = codes1
        self.codes2 = codes2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(self.codes1[idx], self.codes2[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        return {**{k: v.squeeze(0) for k, v in encoding.items()}, 'labels': torch.tensor(self.labels[idx])}

train_dataset = CodePairsDataset(X_train['code1'].values, X_train['code2'].values, y_train.values)
val_dataset = CodePairsDataset(X_test['code1'].values, X_test['code2'].values, y_test.values)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW


# Load CodeBERT with a classification head
model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=2)
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 3
model.train()
for epoch in tqdm(range(num_epochs), desc='Epoch Progress'):
    print(f"Epoch {epoch+1}")
    total_loss = 0
    train_loader = [...]  # Your DataLoader
    for i, batch in enumerate(tqdm(train_loader, desc='Batch Progress', leave=False)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

for epoch in tqdm(range(num_epochs), desc='Epoch Progress'):
    print(f"Epoch {epoch+1}")
    total_loss = 0
    
    for i, batch in enumerate(tqdm(train_loader, desc='Batch Progress', leave=False)):
        print(f"Batch {i}")
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        print(f"Batch loss: {loss.item()}")
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")
model.save_pretrained('tuned-code-bert')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
model.eval()
predictions, true_labels = [], []

for batch in val_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, axis=-1).tolist())
        true_labels.extend(batch['labels'].tolist())

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
print(f"Validation Accuracy: {accuracy}")
# save the model
model.save_pretrained('models/Clonebert')