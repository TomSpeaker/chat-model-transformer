import torch
import json

from model import GPT2Transformer
from CreateTokernizerAndData.tokenizer_custom import build_vocab_from_file, decode

# =====================
# 参数配置
# =====================
train_file = 'train.txt'
data_file = 'train_encoded.jsonl'
model_path = 'saved_models/final_model.pth'  # 最终训练模型
max_len = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 加载词表
# =====================
token2id, id2token = build_vocab_from_file(train_file)
vocab_size = len(token2id)

# =====================
# 加载模型
# =====================
model = GPT2Transformer(
    vocab_size=vocab_size,
    d_model=512,
    num_heads=8,
    num_layers=6,
    max_len=max_len,
    dropout=0.1
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ 模型已成功加载并进入推理模式！")

# =====================
# 推理函数
# =====================
def generate_answer(input_ids):
    input_tensor = torch.tensor(input_ids[:max_len], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_ids = torch.argmax(output, dim=-1).squeeze(0).tolist()
    return pred_ids
