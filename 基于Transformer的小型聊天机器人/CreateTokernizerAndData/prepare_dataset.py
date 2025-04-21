import json
from tokenizer_custom import build_vocab_from_file, encode

# 配置项
max_len = 128
input_file = "train.txt"
output_file = "train_encoded.jsonl"

# 构建词表
token2id, id2token = build_vocab_from_file(input_file)

def process_line(line):
    # 保证是标准格式
    if not line.startswith("<bos>") or "<sep>" not in line or "<eos>" not in line:
        return None

    input_ids = encode(line.strip(), token2id, max_len=max_len)

    # 屏蔽 label 中 <bos> 和 <sep> 前的内容
    sep_index = line.find("<sep>")
    mask_len = len(encode(line[:sep_index+5], token2id, max_len=max_len))  # +5 是因为 <sep> 是 5 个字符

    labels = input_ids.copy()
    for i in range(mask_len):
        labels[i] = -100  # 屏蔽用户部分和特殊符号

    return {"input_ids": input_ids, "labels": labels}

# 处理所有数据
with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        sample = process_line(line)
        if sample:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"✅ 数据处理完成，共生成训练样本，输出文件：{output_file}")
