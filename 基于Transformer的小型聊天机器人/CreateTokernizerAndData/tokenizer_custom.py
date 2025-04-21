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
    while i < len(text):
        if text[i:i+5] in ['<bos>', '<sep>', '<eos>']:
            tokens.append(text[i:i+5])
            i += 5
        else:
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
    text = ''
    for tok in tokens:
        if tok == '<pad>':
            continue
        text += tok
    return text

