import torch
from model import GPT2Transformer
from CreateTokernizerAndData.tokenizer_custom import build_vocab_from_file, encode, decode
import os

# =====================
# 配置参数
# =====================
test_file = 'train.txt'  # 测试文件路径
batch_size = 1  # 测试时批大小通常设为 1
max_len = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "gpt2_qa_model.pth"  # 已训练模型路径

# =====================
# 读取词表
# =====================
token2id, id2token = build_vocab_from_file(test_file)
vocab_size = len(token2id)
print("🚀词表大小：", vocab_size)

# =====================
# 测试数据集
# =====================
class QADataset(torch.utils.data.Dataset):
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
        x = self.samples[idx]  # 直接使用整个样本作为输入
        return x

# =====================
# 加载模型
# =====================
def load_model(model, model_path):
    if os.path.exists(model_path):
        print(f"📦 加载模型参数: {model_path}")
        model.load_state_dict(torch.load(model_path))
        model.eval()  # 设置模型为评估模式
    else:
        print("⚠️ 模型参数文件未找到，无法加载!")

# =====================
# 测试函数
# =====================
def test(model, data_loader, device, id2token):
    print("🚀 开始测试！")
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        for batch_idx, x in enumerate(data_loader):
            x = x.to(device)

            output = model(x)
            output = output.argmax(dim=-1)  # 选择概率最高的 token

            decoded_output = decode(output[0], id2token)
            print(f"Input: {decode(x[0], id2token)}")
            print(f"Predicted Output: {decoded_output}")

# =====================
# 数据加载
# =====================
dataset = QADataset(test_file, token2id, max_len)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

# 加载训练好的模型
load_model(model, model_path)

# =====================
# 开始测试
# =====================
test(model, data_loader, device, id2token)
