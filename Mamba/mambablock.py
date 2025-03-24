from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
# import sys
# sys.path.append('/data/home/yangxiaoda/LaBraM/Mamba')
# from layers.LWT_layers import *
# from layers.RevIN import RevIN
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, c_in, m_layers, d_model, d_ff, dropout, act, d_state, d_conv, m_patch_len):
        super().__init__()

        W_inp_len = m_patch_len
        self.W_P = nn.Linear(W_inp_len, d_model)
        self.m_layers = m_layers

        self.mamba_layers = nn.ModuleList([Mamba_Encoder_Layer(d_model, d_ff, dropout, act, d_state, d_conv, m_patch_len) for i in range(m_layers)])
    
    def forward(self, x):
        bs, nvars, m_patch_len, m_patch_num = x.shape
        x = x.permute(0,1,3,2) 
        x = x.view(bs*nvars, m_patch_num, m_patch_len) 
        # Input encoding
        x = self.W_P(x) # x: [(bs*nvars) x m_patch_num x d_model]

        for i in range(self.m_layers):
            x = self.mamba_layers[i](x)

        x = x.view(bs, nvars, m_patch_num, -1) 
        x = x.permute(0,1,3,2) # x: [bs x nvars x d_model x m_patch_num]

        return x


class Mamba_Encoder_Layer(nn.Module):
    def __init__(self, d_model, d_ff, dropout, act, d_state, d_conv, m_patch_len):
        super().__init__()
        self.mamba = Mamba(d_model, d_state=d_state, d_conv=d_conv)
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu if act == "relu" else F.gelu

    def forward(self, x):
        x = self.mamba(x)
        x = self.lin2(self.act(self.lin1(x)))

        return x
    