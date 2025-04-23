from CreateTokernizerAndData.tokenizer_custom import load_vocab_from_file, encode,decode
import torch
from model import GPT2Transformer

# ===================== 参数配置 =====================
vocab_file = 'vocab.json'  # 用于构建词表的文件
model_path = 'saved_models/final_model.pth'
max_len = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================== 加载词表 =====================
token2id, id2token = load_vocab_from_file(vocab_file)
vocab_size = len(token2id)

# ===================== 加载模型 =====================
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

# ===================== 示例推理 =====================
# 示例输入
# 
while True:
    question = input("请输入问题:")
    question = '<bos>'+question+'<eos>'
    # 编码输入
    input_ids = encode(question, token2id, max_len=max_len)  # List[int]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]

    # 解码原始输入
    decoded_input = decode(input_ids, id2token)

    # 模型推理
    with torch.no_grad():
        outputs = model(input_tensor)  # 输出形状: [1, seq_len, vocab_size]
        predictions = torch.argmax(outputs, dim=-1)  # [1, seq_len]
        predicted_ids = predictions[0].tolist()  # List[int]

    # 解码模型输出
    decoded_output = decode(predicted_ids, id2token)
    # ===================== 打印结果 =====================
    print(f"🟡 原始问题: {question}")
    print(f"🟠 模型输出文本: {decoded_output}")
    