import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
import json

from model import GPT2Transformer
from CreateTokernizerAndData.tokenizer_custom import build_vocab_from_file, encode, decode

# =====================
# 配置参数
# =====================
train_file = 'train.txt'  # 用于构建词表的文件
train_data_file = 'train_encoded.jsonl'  # 训练数据文件
batch_size = 16
max_len = 128
num_epochs = 200
learning_rate = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "gpt2_qa_model.pth"

# =====================
# 读取词表
# =====================
token2id, id2token = build_vocab_from_file(train_file)
vocab_size = len(token2id)
print("🚀 词表大小：", vocab_size)

# =====================
# 自定义数据集
# =====================
import torch
import json
from CreateTokernizerAndData.tokenizer_custom import decode

# 自定义数据集
class QADataset(Dataset):
    def __init__(self, filepath, token2id, max_len):
        self.samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())  # 解析每一行 JSON 数据
                input_ids = sample['input_ids']
                labels = sample['labels']

                # 如果需要截断或填充到最大长度
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]

                # 填充或处理标签，避免出现 -100 的标签
                while len(input_ids) < max_len:
                    input_ids.append(token2id['<pad>'])  # 使用 <pad> 填充
                    labels.append(-100)  # 标签对应 -100，表示忽略这个位置

                self.samples.append((torch.tensor(input_ids), torch.tensor(labels)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x, y

# 加载数据集
dataset = QADataset(train_data_file, token2id, max_len)

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
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # 每 50 epoch 学习率减半

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
    if model_path:
        load_model(model, model_path)

    print("🚀 开始训练！")
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)
        scheduler.step()

        print(f"📅 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % 50 == 0 and epoch != 0:
            torch.save(model.state_dict(), model_path)
            print(f"💾 Epoch {epoch+1} 模型已保存")

    # 画 loss 曲线
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    print("📉 Loss 曲线已保存为 loss_curve.png")

# =====================
# 开始训练
# =====================
train(model, data_loader, criterion, optimizer, num_epochs, model_path=model_path)

# =====================
# 最终保存
# =====================
torch.save(model.state_dict(), model_path)
print(f"✅ 最终模型已保存为 {model_path}")
