import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import clones

def rotate_half(x):
    # 将输入张量的后半部分取反，用于实现旋转
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(tensor, sin, cos):
    # 应用旋转位置编码：tensor * cos + rotate_half(tensor) * sin
    return (tensor * cos.unsqueeze(1)) + (rotate_half(tensor) * sin.unsqueeze(1))

class RoPEAttention(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        # 预计算频率因子 theta_i
        theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('theta', theta)
        
        self.mz = None
        self.IS_UPDATE = False
        
    def update_mz(self, mz):
        self.mz = mz
        self.IS_UPDATE = True
        
    def reset(self):
        self.mz = None
        self.IS_UPDATE = False
        
    def get_rotary_matrix(self, seq_len):
        # 生成位置m的sin和cos值
        # m = torch.arange(seq_len, device=self.theta.device)
        m = self.mz
        # 构造频率矩阵：m * theta
        freqs = torch.einsum('ij,k->ijk', m, self.theta)
        # 生成sin和cos
        sin, cos = torch.sin(freqs), torch.cos(freqs)
        # 扩展维度以匹配输入形状 [seq_len, d_model]
        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)
        return sin, cos
    
    def forward(self, q, k):
        assert self.IS_UPDATE == True, "Please update mz first."
        
        batch_size, head, seq_len, d_model = q.shape
        sin, cos = self.get_rotary_matrix(seq_len)
        # 应用旋转位置编码
        
        q_rot = apply_rotary_pos_emb(q, sin, cos)
        k_rot = apply_rotary_pos_emb(k, sin, cos)
        # 计算注意力得分
        scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))
        return scores
    

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "传入head个数及model的维度."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 这里假设d_v=d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
        self.rope = RoPEAttention(d_model//h)
    
        
    def forward(self, query, key, value, mask=None):
        
        if mask is not None:
            # 相同的mask适应所有的head.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) 首先使用线性变换，然后把d_model分配给h个Head，每个head为d_k=d_model/h         
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) 使用attention函数计算scaled-Dot-product-attention 
        x, self.attn = self.attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) 实现Multi-head attention，用view函数把8个head的64维向量拼接成一个512的向量。
        #然后再使用一个线性变换(512,521)，shape不变. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        
        
        return self.linears[-1](x)
    
    
    def attention(self, query, key, value, mask=None, dropout=None):
        
        scores = self.rope(query, key)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn