import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
import json

from model import GPT2Transformer
from CreateTokernizerAndData.tokenizer_custom import load_vocab_from_file, encode, decode

# =====================
# 配置参数
# =====================
vocab_file = 'vocab.json'
train_data_file = 'train_encoded_v2.jsonl'
batch_size = 16
max_len = 128
num_epochs = 50
learning_rate = 1e-4
epoch_save = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)
model_path = os.path.join(model_save_dir, "final_model.pth")
log_file = os.path.join(model_save_dir, "training_log.txt")
state_file = os.path.join(model_save_dir, "train_state.json")

# =====================
# 读取词表
# =====================
token2id, id2token = load_vocab_from_file(vocab_file)
vocab_size = len(token2id)
print("🚀 词表大小：", vocab_size)

# =====================
# 自定义数据集
# =====================
class QADataset(Dataset):
    def __init__(self, filepath, token2id, max_len):
        self.samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                input_ids = sample['input_ids'][:max_len]
                labels = sample['labels'][:max_len]

                while len(input_ids) < max_len:
                    input_ids.append(token2id['<pad>'])
                    labels.append(-100)

                self.samples.append((torch.tensor(input_ids), torch.tensor(labels)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# =====================
# 加载数据集
# =====================
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
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

# =====================
# 模型 & 状态加载与保存
# =====================
def load_model(model, model_path):
    if os.path.exists(model_path):
        print(f"📦 加载模型参数: {model_path}")
        model.load_state_dict(torch.load(model_path))
        return True
    else:
        print("⚠️ 未找到模型参数，将从头开始训练")
        return False

def load_train_state():
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        print(f"📄 已加载训练状态：轮次 {state['epoch']}, 学习率 {state['lr']}")
        return state["epoch"], state["lr"]
    return 0, learning_rate

def save_train_state(epoch, lr):
    state = {
        "epoch": epoch,
        "lr": lr
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f)

# =====================
# 训练函数
# =====================
def train(model, data_loader, criterion, optimizer, num_epochs, model_path=None):
    loaded = load_model(model, model_path) if model_path else False
    start_epoch, resumed_lr = load_train_state()

    for param_group in optimizer.param_groups:
        param_group['lr'] = resumed_lr

    if loaded:
        print("🔄 继续训练模型")
    else:
        print("🆕 开始首次训练")

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n我对 GPT2Transformer 模型进行了训练，共计 {num_epochs} 轮，开始训练。\n")

    model.train()
    loss_history = []

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
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

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"轮次 {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - 学习率: {scheduler.get_last_lr()[0]:.6f}\n")

        if (epoch + 1) % epoch_save == 0:
            save_name = f"epoch_{epoch+1:03d}.pth"
            save_path = os.path.join(model_save_dir, save_name)
            torch.save(model.state_dict(), save_path)
            save_train_state(epoch + 1, scheduler.get_last_lr()[0])
            print(f"💾 模型保存到 {save_path}")

# =====================
# 开始训练
# =====================
train(model, data_loader, criterion, optimizer, num_epochs, model_path=model_path)

# =====================
# 最终保存模型
# =====================
torch.save(model.state_dict(), model_path)
print(f"✅ 最终模型已保存为 {model_path}")
