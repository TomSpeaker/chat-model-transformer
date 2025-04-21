import json

# 输入和输出文件路径
input_file = 'qa_final.json'
output_file = 'train.txt'

# 特殊标记
BOS = '<bos>'
SEP = '<sep>'
EOS = '<eos>'

# 加载 JSON 数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 存放格式化数据
output_lines = []

# 遍历每个对话
for dialogue in data:
    turns = dialogue.get("turns", [])
    for i in range(len(turns) - 1):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "ai":
            prompt = turns[i]["text"].strip().replace("\n", " ")
            response = turns[i + 1]["text"].strip().replace("\n", " ")
            line = f"{BOS}{prompt}{SEP}{response}{EOS}"
            output_lines.append(line)

# 写入处理后的训练数据
with open(output_file, 'w', encoding='utf-8') as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"已提取 {len(output_lines)} 条问答对，保存至 {output_file}")
