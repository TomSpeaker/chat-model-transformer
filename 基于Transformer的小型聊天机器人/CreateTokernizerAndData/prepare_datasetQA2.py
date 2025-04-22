import json
from tokenizer_custom import load_vocab_from_file, encode

# ========= 配置项 =========
max_len = 128
input_file = "train.txt"
output_file = "train_encoded_v2.jsonl"
vocab_file = "vocab.json"

# ========= 特殊 Token =========
BOS_TOKEN = "<bos>"
SEP_TOKEN = "<sep>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

# ========= 加载词表 =========
token2id, id2token = load_vocab_from_file(vocab_file)
print(f"✅ 词表读取成功，词表大小: {len(token2id)}")

# 添加特殊 token（如果缺失）
special_tokens = [BOS_TOKEN, SEP_TOKEN, EOS_TOKEN, PAD_TOKEN]
for token in special_tokens:
    if token not in token2id:
        token2id[token] = len(token2id)
        id2token[len(id2token)] = token

# 获取特殊 token 的 ID
bos_token_id = token2id[BOS_TOKEN]
sep_token_id = token2id[SEP_TOKEN]
eos_token_id = token2id[EOS_TOKEN]
pad_token_id = token2id[PAD_TOKEN]

# ========= 核心函数：处理单行 =========
def process_line(line):
    line = line.strip()

    if not line.startswith(BOS_TOKEN) or SEP_TOKEN not in line or EOS_TOKEN not in line:
        return None  # 格式不合法

    try:
        # 拆分为 question 和 answer
        question_part, answer_part = line.split(SEP_TOKEN, 1)
        question = question_part.strip()
        answer = answer_part.strip()

        # 自动纠正特殊 token（防止缺失）
        if not question.startswith(BOS_TOKEN):
            question = BOS_TOKEN + question
        if not question.endswith(EOS_TOKEN):
            question += EOS_TOKEN
        if not answer.startswith(BOS_TOKEN):
            answer = BOS_TOKEN + answer
        if not answer.endswith(EOS_TOKEN):
            answer += EOS_TOKEN

        # 编码
        input_ids = encode(question, token2id)
        labels = encode(answer, token2id)

        # 如果 input_ids 或 labels 中没有 <eos>，则丢弃该条数据
        if eos_token_id not in input_ids or eos_token_id not in labels:
            return None

        # 填充到 max_len
        input_ids += [pad_token_id] * (max_len - len(input_ids))
        labels += [pad_token_id] * (max_len - len(labels))

        return {"input_ids": input_ids, "labels": labels}

    except Exception as e:
        print(f"❌ 处理出错: {e}")
        return None

# ========= 执行处理 =========
sample_count = 0
with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        sample = process_line(line)
        if sample:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            sample_count += 1

print(f"🎉 数据处理完成，共生成 {sample_count} 条训练样本，输出文件：{output_file}")
