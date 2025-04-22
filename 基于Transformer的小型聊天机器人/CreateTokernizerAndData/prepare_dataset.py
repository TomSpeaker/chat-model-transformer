import json
from tokenizer_custom import build_vocab_from_file, encode

# 配置项
max_len = 128
input_file = "train.txt"
output_file = "train_encoded.jsonl"

# 特殊 token ID 获取
BOS_TOKEN = "<bos>"
SEP_TOKEN = "<sep>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"  # 假设填充 token 为 <pad>

# 构建词表
token2id, id2token = build_vocab_from_file(input_file)

# 确保特殊 token 存在
special_tokens = [BOS_TOKEN, SEP_TOKEN, EOS_TOKEN, PAD_TOKEN]

# 检查并添加特殊 token
for token in special_tokens:
    if token not in token2id:
        token2id[token] = len(token2id)  # 给特殊 token 分配一个新的 ID
        id2token[len(id2token)] = token  # 反向映射

# 获取特殊 token 的 ID
bos_token_id = token2id[BOS_TOKEN]
sep_token_id = token2id[SEP_TOKEN]
eos_token_id = token2id[EOS_TOKEN]
pad_token_id = token2id[PAD_TOKEN]

def process_line(line):
    # 保证是标准格式
    if not line.startswith(BOS_TOKEN) or SEP_TOKEN not in line or EOS_TOKEN not in line:
        return None

    # 编码整行文本
    input_ids = encode(line.strip(), token2id, max_len=max_len)

    # 处理填充：如果长度不足 max_len，则填充到 max_len
    if len(input_ids) < max_len:
        input_ids += [pad_token_id] * (max_len - len(input_ids))

    # 在 token 序列中查找 <sep> 的 token 位置
    try:
        sep_idx = input_ids.index(sep_token_id)
    except ValueError:
        return None  # 如果找不到 <sep>，跳过

    # 构建 labels，<sep> 之前（包含）的位置设为 0，其余正常训练
    labels = input_ids.copy()

    # 如果 labels 的第一位不是 BOS_TOKEN，则将其设置为 BOS_TOKEN
    if labels[0] != bos_token_id:
        labels[0] = bos_token_id

    for i in range(sep_idx + 1):  # 包含 <sep>
        labels[i] = 0  # 使用 0 来表示填充部分

    # 对填充部分的 labels 设置为 0
    for i in range(len(input_ids), max_len):
        labels[i] = 0

    # 检查 <sep> 后是否有有效内容
    valid_content_after_sep = any(label != 0 for label in labels[sep_idx + 1:])
    
    if not valid_content_after_sep:
        return None  # 如果 <sep> 后没有有效内容，则跳过

    return {"input_ids": input_ids, "labels": labels}

# 处理所有数据
sample_count = 0
with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        sample = process_line(line)
        if sample:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            sample_count += 1

print(f"✅ 数据处理完成，共生成 {sample_count} 条训练样本，输出文件：{output_file}")
