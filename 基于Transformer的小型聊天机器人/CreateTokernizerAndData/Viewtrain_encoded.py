import json
from tokenizer_custom import build_vocab_from_file

# 重新加载词表
input_file = "train.txt"
token2id, id2token = build_vocab_from_file(input_file)

# 定义解码函数
def decode_ids(ids):
    """将 token ids 转换回文本"""
    return [id2token.get(i, "<unk>") for i in ids]

# 读取并解码数据
sample_count = 0
with open("train_encoded.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        
        input_ids = data["input_ids"]
        labels = data["labels"]
        
        # 解码 input_ids 和 labels
        input_text = decode_ids(input_ids)
        labels_text = decode_ids(labels)
        
        # 输出解码后的文本
        print(f"样本 {sample_count + 1}")
        print("输入文本 (input_ids):")
        print(" ".join(input_text))
        print("\n标签文本 (labels):")
        print(" ".join(labels_text))
        print("-" * 50)
        
        sample_count += 1
        if sample_count == 10:  # 只打印前10个样本
            break
