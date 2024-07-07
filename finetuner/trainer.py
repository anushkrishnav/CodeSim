import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from finetuner.tokenizer import CodePairsDataset
from transformers import RobertaForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm


def main():
    columns = ["code1", "code2", "similar"]
    df = pd.read_parquet('data/train/clone-detection-600k-5fold.parquet',columns=columns)

    df = df[columns]
    df = df.dropna()
    kf = KFold(n_splits=5)
    num_epochs = 10
    device = "cpu"  # Assuming training on CPU
    for fold, (train_index, val_index) in enumerate(kf.split(df)):
        train_df = df.iloc[train_index]
        train_loader = CodePairsDataset.get_loader(train_df, ['code1', 'code2'], 'similar')
        model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=2)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-5)
        model.train()
        # val_df = df.iloc[val_index]
        # val_loader = CodePairsDataset.get_loader(val_df, ['code1', 'code2'], 'similar')
    for epoch in tqdm(range(num_epochs), desc='Epoch Progress'):
        total_loss = 0
        
        for batch in tqdm(train_loader, desc='Batch Progress', leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Loss: {avg_loss}")
    model.save_pretrained('models/Clonebert/tuned-code-bert')


if __name__ == '__main__':
    main()
