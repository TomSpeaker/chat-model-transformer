import torch
import json

from model import GPT2Transformer
from CreateTokernizerAndData.tokenizer_custom import build_vocab_from_file, decode, encode_question

# =====================
# 参数配置
# =====================
train_file = 'train.txt'
model_path = 'saved_models/final_model.pth'
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

# model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ 模型已成功加载并进入推理模式！")
print("请输入问题（输入 q 退出）：")

# =====================
# 推理函数
# =====================
while True:
    question = input("📝 问题: ")
    if question.strip().lower() in ['q', 'quit', 'exit']:
        print("👋 推理结束，欢迎再次使用！")
        break

    # 编码输入
    input_ids = encode_question(question, token2id, max_len)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 模型推理
    with torch.no_grad():
        output = model(input_tensor)

    # 取 logits 中 <sep> 后面的预测结果
    output_ids = torch.argmax(output, dim=-1)[0].tolist()

    # 查找 <sep> 的位置，跳过问题部分，仅显示回答内容
    try:
        sep_index = input_ids.index(token2id["<sep>"])
        predicted_answer_ids = output_ids[sep_index + 1:]
    except ValueError:
        predicted_answer_ids = output_ids  # 如果没有<sep>，全部展示

    # 解码输出
    answer = decode(predicted_answer_ids, id2token)
    answer = answer.split("<eos>")[0]  # 截断至 <eos>

    print("🤖 答案:", answer.strip())
