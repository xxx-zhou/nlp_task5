import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import numpy as np

# 数据集类
class PoetryDataset(Dataset):
    def __init__(self, text, tokenizer, max_len=128):
        self.text = text
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text) - 1

    def __getitem__(self, idx):
        x = self.text[idx:idx+1]
        y = self.text[idx+1:idx+2]
        x_ids = self.tokenizer.encode(x, add_special_tokens=False)
        y_ids = self.tokenizer.encode(y, add_special_tokens=False)
        return torch.tensor(x_ids, dtype=torch.long), torch.tensor(y_ids, dtype=torch.long)

# 初始化BERT和Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=False)

# 修改BERT的输出层
model.classifier = nn.Linear(model.config.hidden_size, len(tokenizer))

# 数据准备
with open("poetryFromTang.txt", "r", encoding="utf-8") as file:
    text = file.read()

dataset = PoetryDataset(text, tokenizer)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, len(tokenizer)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# 计算困惑度
def calculate_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    total_words = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, len(tokenizer)), targets.view(-1))
            total_loss += loss.item()
            total_words += targets.size(0)
    return torch.exp(torch.tensor(total_loss / total_words))

perplexity = calculate_perplexity(model, dataloader)
print(f"Perplexity: {perplexity}")