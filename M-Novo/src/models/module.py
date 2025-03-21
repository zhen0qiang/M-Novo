import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import math

def clones(module, N):
    "克隆N个完全相同的SubLayer，使用了copy.deepcopy"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Encoder是N个EncoderLayer的堆积而成"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        #layer是一个SubLayer，我们clone N个
        self.layers = clones(layer, N)
        #再加一个LayerNorm层
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "把输入(x,mask)被逐层处理"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) #N个EncoderLayer处理完成之后还需要一个LayerNorm

class EncoderLayer(nn.Module):
    "Encoder由self-attn and feed forward构成"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "如上图所示"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class LayerNorm(nn.Module):
    "构建一个layernorm模型"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    LayerNorm + sublayer(Self-Attenion/Dense) + dropout + 残差连接
    为了简单，把LayerNorm放到了前面，这和原始论文稍有不同，原始论文LayerNorm在最后
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        #将残差连接应用于具有相同大小的任何子层
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "实现PE函数"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach()
        return self.dropout(x)

class Generator(nn.Module):
    """定义标准的一个全连接（linear）+ softmax
    根据Decoder的隐状态输出一个词
    d_model是Decoder输出的大小，vocab是词典大小"""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    #全连接再加上一个softmax
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
