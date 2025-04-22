import json
from collections import Counter

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
# Step 2: 保存词表到文件
# =============================
def save_vocab_to_file(token2id, id2token, filepath):
    vocab_data = {
        'token2id': token2id,
        'id2token': {str(k): v for k, v in id2token.items()}  # key 转字符串，避免 JSON 问题
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=4)
    print(f"✅ 词表已保存到 {filepath}")

# =============================
# Step 3: 从文件加载词表
# =============================
def load_vocab_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    token2id = vocab_data['token2id']
    id2token = {int(k): v for k, v in vocab_data['id2token'].items()}  # key 转回整数
    print(f"📂 词表已从 {filepath} 加载")
    return token2id, id2token

# =============================
# Step 4: 编码函数
# =============================
def encode(text, token2id, max_len=128):
    # 添加 <bos> 和 <eos>
    tokens = ['<bos>'] + list(text) + ['<eos>']

    # 转成 ID
    ids = [token2id.get(tok, token2id['<unk>']) for tok in tokens]

    # 补齐或截断
    ids = ids[:max_len]
    ids += [token2id['<pad>']] * (max_len - len(ids))

    return ids

# =============================
# Step 5: 解码函数
# =============================
def decode(ids, id2token):
    tokens = [id2token.get(i, '<unk>') for i in ids]

    # 过滤掉特殊符号
    text = ''.join([tok for tok in tokens if tok not in ['<pad>', '<unk>', '<bos>', '<eos>']])
    return text

# =============================
# 示例使用
# =============================
if __name__ == '__main__':
    train_file = 'train.txt'      # 构建词表的数据
    vocab_file = 'vocab.json'     # 词表保存位置

    # 1. 构建词表
    token2id, id2token = build_vocab_from_file(train_file)

    # 2. 保存词表
    save_vocab_to_file(token2id, id2token, vocab_file)

    # 3. 加载词表
    loaded_token2id, loaded_id2token = load_vocab_from_file(vocab_file)

    # 4. 测试编码 & 解码
    question = "你好"
    encoded = encode(question, loaded_token2id, max_len=40)
    decoded = decode(encoded, loaded_id2token)

    print(f"\n🟢 原始问题: {question}")
    print(f"🔢 编码后的前几个 token: {encoded[:10]}...")
    print(f"🔤 解码后的问题: {decoded}")
