import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import math
import time

from .module import *
from .attn import *

class Transfomer(nn.Module):
    """
    这是一个标准的 Encoder-Decoder架构
    """
    def __init__(self, encoder, src_embed, generator):
        super(Transfomer, self).__init__()
        # encoder和decoder都是构造的时候传入的，这样会非常灵活
        self.encoder = encoder

        # 输入和输出的embedding
        self.src_embed = src_embed  

        self.generator = generator
        
    def forward(self, src, src_mask=None):
        #接收并处理屏蔽src和目标序列，
        #首先调用encode方法对输入进行编码，然后调用decode方法进行解码
        x = self.encode(src, src_mask)
        
        return self.generator(x)
    
    def encode(self, src, src_mask):
        #传入参数包括src的embedding和src_mask
        return self.encoder(self.src_embed(src), src_mask)
    
    def update_mz(self, mz):
        for encoder_layer in self.encoder.layers:
            encoder_layer.self_attn.rope.update_mz(mz)
        
        self.src_embed[1].update_mz(mz)

def make_model(src_vocab,tgt_vocab,N=6,d_model=512, d_ff=2048, h=8, dropout=0.1):
    "构建模型"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transfomer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # 随机初始化参数，这非常重要用Glorot/fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
