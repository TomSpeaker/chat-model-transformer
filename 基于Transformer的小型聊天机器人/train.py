import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import GPT2Transformer
from CreateTokernizerAndData.tokenizer_custom import build_vocab_from_file, encode, decode

import os

# =====================
# 配置参数
# =====================
train_file = 'train.txt'
batch_size = 16
max_len = 128
num_epochs = 10
learning_rate = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "gpt2_qa_model.pth"  # 保存模型的路径

# =====================
# 读取词表
# =====================
token2id, id2token = build_vocab_from_file(train_file)
vocab_size = len(token2id)
print("🚀词表大小：",vocab_size)

# =====================
# 自定义数据集
# =====================
class QADataset(Dataset):
    def __init__(self, filepath, token2id, max_len):
        self.samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                ids = encode(line, token2id, max_len)
                self.samples.append(torch.tensor(ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx][:-1]  # 输入
        y = self.samples[idx][1:]   # 目标（下一个 token）
        return x, y

# =====================
# 数据加载
# =====================
dataset = QADataset(train_file, token2id, max_len)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# =====================
# 构建模型
# =====================
model = GPT2Transformer(
    vocab_size=vocab_size,
    d_model=512,
    num_heads=8,
    num_layers=6,
    max_len=max_len,
    dropout=0.1
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"🚀 当前模型的总参数量: {total_params:,}")
criterion = nn.CrossEntropyLoss(ignore_index=token2id['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =====================
# 加载模型参数的函数
# =====================
def load_model(model, model_path):
    if os.path.exists(model_path):
        print(f"📦 加载模型参数: {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("⚠️ 模型参数文件未找到，无法加载!")

# =====================
# 训练函数
# =====================
def train(model, data_loader, criterion, optimizer, num_epochs, model_path=None):
    # 仅加载模型参数
    if model_path:
        load_model(model, model_path)
    
    print("🚀 开始训练！")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)  # [batch, seq_len, vocab_size]
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"📅 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

        # 每个 epoch 保存一次模型
        torch.save(model.state_dict(), model_path)
        print(f"💾 Epoch {epoch+1} 模型已保存")

# =====================
# 开始训练或加载并继续训练
# =====================
train(model, data_loader, criterion, optimizer, num_epochs, model_path=model_path)

# =====================
# 保存最终模型
# =====================
torch.save(model.state_dict(), model_path)
print(f"✅ 最终模型已保存为 {model_path}")
