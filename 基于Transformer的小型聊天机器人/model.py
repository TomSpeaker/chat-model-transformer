import torch
import torch.nn as nn
import torch.nn.functional as F

# 参数配置
vocab_size = 30522  # 词汇表大小，使用预训练模型时需要根据实际情况调整
d_model = 512  # 嵌入维度
num_heads = 8  # 多头注意力头数
num_layers = 6  # Transformer 层数
max_len = 512  # 最大序列长度
dropout = 0.1  # Dropout 概率

# 位置编码（Position Encoding）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()

# 自注意力层（Self-Attention Layer）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0  # 确保能均匀分配
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # 线性映射后分头
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算 QK^T
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn_weights, dim=-1)
        
        # 计算输出
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.out_linear(output)

# 前馈神经网络（Feed-Forward Network）
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# Transformer 层（Transformer Layer）
class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))  # 残差连接 + LayerNorm
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))  # 残差连接 + LayerNorm
        
        return x

# Transformer 模型
class GPT2Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len, dropout=0.1):
        super(GPT2Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.linear_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)  # 输入嵌入
        x = self.positional_encoding(x)  # 加上位置编码

        for layer in self.layers:
            x = layer(x, mask)
        
        output = self.linear_out(x)
        return output

# # 创建模型
# model = GPT2Transformer(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, max_len=max_len)

# # 输入数据样例
# input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 假设这是一个已经编码的输入
# output = model(input_ids)

# print(output.shape)  # 输出形状
