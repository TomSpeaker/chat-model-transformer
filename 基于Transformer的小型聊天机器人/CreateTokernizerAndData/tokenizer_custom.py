import json
from collections import Counter
import os

# =============================
# Step 1: 构建词表
# =============================
def build_vocab_from_file(filename, min_freq=1):
    counter = Counter()

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            for ch in line:
                counter[ch] += 1

    # 特殊符号
    special_tokens = ['<pad>', '<unk>', '<bos>', '<sep>', '<eos>']
    vocab = special_tokens + [token for token, freq in counter.items() if freq >= min_freq]

    token2id = {token: idx for idx, token in enumerate(vocab)}
    id2token = {idx: token for token, idx in token2id.items()}

    print(f"📘 词表大小: {len(vocab)}")
    return token2id, id2token

# =============================
# Step 2: 编码函数
# =============================
def encode(text, token2id, max_len=128):
    tokens = []
    i = 0
    special_tokens = ['<bos>', '<sep>', '<eos>']
    while i < len(text):
        matched = False
        for token in special_tokens:
            if text[i:i+len(token)] == token:
                tokens.append(token)
                i += len(token)
                matched = True
                break
        if not matched:
            tokens.append(text[i])
            i += 1

    ids = [token2id.get(tok, token2id['<unk>']) for tok in tokens]
    ids = ids[:max_len]
    ids += [token2id['<pad>']] * (max_len - len(ids))
    return ids


# =============================
# Step 3: 解码函数
# =============================
def decode(ids, id2token):
    tokens = [id2token.get(i, '<unk>') for i in ids]
    # 去除 <pad>，并确保格式正确
    text = ''.join([tok for tok in tokens if tok != '<pad>'])
    return text

# filename = 'train.txt'

# # Step 1: 构建词表
# token2id, id2token = build_vocab_from_file(filename)


# # Step 2: 读取train.txt并编码前十个句子
# with open(filename, 'r', encoding='utf-8') as f:
#     lines = f.readlines()

# Step 3: 编码并解码前十个句子
# for i in range(min(1, len(lines))):  # 获取前十行或文件中的所有行
#     line = lines[i].strip()

#     # 编码
#     encoded = encode(line, token2id)
#     # 解码
#     decoded = decode(encoded, id2token)

#     # 输出原文、编码后的前几个token和解码后的结果
#     print(f"原文: {line}")
#     print(f"编码后的前几个结果: {encoded[:10]}...")  # 只显示编码后的前10个
#     print(f"解码后的结果: {decoded}")
#     print("=" * 50)

def encode_question(question: str, token2id: dict, max_len: int = 128) -> list:
    """
    给定问题内容，生成加上 <bos> 和 <sep> 的编码序列。
    """
    special_tokens = ["<bos>", "<sep>"]
    input_text = special_tokens[0] + question + special_tokens[1]

    # 把每个字符作为 token 进行编码
    input_ids = [token2id.get(char, token2id.get("<unk>", 1)) for char in input_text]

    # 补齐到 max_len
    if len(input_ids) < max_len:
        input_ids += [token2id.get("<pad>", 0)] * (max_len - len(input_ids))
    else:
        input_ids = input_ids[:max_len]

    return input_ids
