# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import math
import torch
import torch.nn as nn
from functools import partial
from Mamba.mambablock import MambaBlock
from modeling_finetune import Block, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange
import numpy as np
import torch.nn.functional as F
def qprint(var,str):
    print("\033[92m"+"{}:{}".format(str,var)+"\033[0m")

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class TemporalConv(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x


class NeuralTransformerForMaskedEEGModeling(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = TemporalConv(out_chans=out_chans)
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))# 分类标记 (CLS token)，初始化为全零
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))# Mask标记，用于替换遮掩的输入

        if use_abs_pos_emb:# 如果使用绝对位置编码
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))# 位置编码 (含CLS token)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)# 时间嵌入，支持时间维度的信息
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([# 创建多个Transformer块
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.time_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()# 修正初始化权重

    def fix_init_weight(self):
        def rescale(param, layer_id):# 修正权重，使用 layer_id 的平方根进行缩放
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):# 对每个Transformer块的注意力权重和MLP权重进行缩放
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, input_chans, bool_masked_pos): # 提取特征的主函数
        
        qprint(input_chans,'input_chans')
        batch_size, c, time_window, _ = x.size()
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)   # 复制CLS标记
        mask_token = self.mask_token.expand(batch_size, seq_len, -1) # 复制Mask标记

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token) # 将Mask位转换为权重
        x = x * (1 - w) + mask_token * w # 替换被Mask掉的Patch

        x = torch.cat((cls_tokens, x), dim=1)
        qprint(x.shape,'x.shape')
        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        # qprint(pos_embed_used.shape,'pos_embed_used.shape')
        if self.pos_embed is not None: # 处理位置编码
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            qprint(pos_embed.shape,'pos_embed.shape')
            x = x + pos_embed
        if self.time_embed is not None:# 处理时间编码
            time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        #这之后的x的shape应该是什么样的
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:#B,seqlen,dim
            # qprint(x.shape,'front:x.shape')
            x = blk(x, rel_pos_bias=rel_pos_bias)# 依次通过每个Transformer块
            # qprint(x.shape,'back:x.shape')

        return self.norm(x)# 输出归一化后的特征

    def forward(self, x, input_chans=None, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False, return_all_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)# 如果没有传入 Mask 的布尔矩阵，则默认所有位置都未被 Mask
        x = self.forward_features(x, input_chans=input_chans, bool_masked_pos=bool_masked_pos)# 使用 `forward_features` 提取特征
        # from IPython import embed; embed()
        if return_all_patch_tokens:# 如果需要返回所有 Patch 特征，直接返回
            return x
        x = x[:, 1:]# 丢弃第一个位置的 [CLS] Token
        
        if return_patch_tokens:# 如果只需要 Patch Tokens 的特征，返回结果
            return x
        if return_all_tokens:# 如果需要对所有 Tokens 应用输出层，直接返回语言建模头的输出
            return self.lm_head(x)
        else:
            # return the masked tokens
            return self.lm_head(x[bool_masked_pos])# 返回被 Mask 的位置对应的输出
    
    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)# 如果没有传入 Mask 的布尔矩阵，则默认所有位置都未被 Mask
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)# 使用 Patch 嵌入层提取初步特征
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 复制 [CLS] Token 和 Mask Token
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked EEG tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)# 拼接 [CLS] Token
        if self.pos_embed is not None:
            x = x + self.pos_embed# 添加位置编码
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)# 除最后一层外，逐层通过 Transformer Block
            else:
                # with torch.cuda.amp.autocast(enabled=False):
                x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)# 最后一层计算 QKV 输出

        if split_out_as_qkv:# 如果需要分别返回 Q, K, V
            x = self.norm(x)
            x = self.lm_head(x) # [b, n+1, 3*c]
            q, k, v = x.chunk(3, dim=-1) # [b, n+1, c]
            b, n, c =q.shape
            q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)# 重塑 Q, K, V 为 [B, num_heads, seq_len, head_dim] 
            k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            return x, q, k, v
        else:
            x = self.norm(x)
            x = x[:, 1:]
            x = self.lm_head(x[bool_masked_pos])

            q, k, v = qkv[0], qkv[1], qkv[2]

        return x, q, k, v

    def get_last_selfattention(self, x):
        x = self.patch_embed(x)# 使用 Patch 嵌入层提取初步特征
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)# 复制 [CLS] Token
        x = torch.cat((cls_tokens, x), dim=1) # 拼接 [CLS] Token
        if self.pos_embed is not None:
            x = x + self.pos_embed # 添加位置编码
        x = self.pos_drop(x) # 应用 Dropout
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias) # 除最后一层外，逐层通过 Transformer Block
            else:
                # return attention of the last block
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)# 返回最后一层的注意力权重
            

class NeuralTransformerForMEM(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.student = NeuralTransformerForMaskedEEGModeling(EEG_size, patch_size, in_chans, out_chans, vocab_size, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, attn_head_dim,
                 use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std)
        
        self.lm_head = nn.Linear(embed_dim, vocab_size)# 语言建模头
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.ReLU()
        )

        trunc_normal_(self.lm_head.weight, std=init_std)# 初始化语言建模头权重
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'student.cls_token', 'student.pos_embed', 'student.time_embed'}
    
    def forward(self, x, input_chans=None, bool_masked_pos=None):
        # from IPython import embed; embed()
        x_masked = self.student(x, input_chans, bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls = x_masked[:, 1:]# 丢弃 [CLS] Token
        qprint(bool_masked_pos.shape,'bool_masked_pos.shape')
        qprint(x_masked_no_cls.shape,'x_masked_no_cls.shape')
        x_rec = self.lm_head(x_masked_no_cls[bool_masked_pos])# 重建被 Mask 的位置

        #symetric 对称 Mask：处理 Mask 和未 Mask 的情况
        x_masked_sym = self.student(x, input_chans, ~bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls_sym = x_masked_sym[:, 1:]
        x_rec_sym = self.lm_head(x_masked_no_cls_sym[~bool_masked_pos])

        return x_rec, x_rec_sym # 返回两种重建结果


@register_model
def labram_base_patch200_1600_8k_vocab(pretrained=False, **kwargs): #5M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']# 提取词汇表大小
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = NeuralTransformerForMEM(# 初始化 NeuralTransformerForMEM 模型
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def labram_large_patch200_1600_8k_vocab(pretrained=False, **kwargs): #50M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = NeuralTransformerForMEM(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def labram_huge_patch200_1600_8k_vocab(pretrained=False, **kwargs): #380M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = NeuralTransformerForMEM(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=32,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

























class GT_NeuralTransformerForMaskedEEGModeling(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = TemporalConv(out_chans=out_chans)
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))# 分类标记 (CLS token)，初始化为全零
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))# Mask标记，用于替换遮掩的输入

        if use_abs_pos_emb:# 如果使用绝对位置编码
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))# 位置编码 (含CLS token)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)# 时间嵌入，支持时间维度的信息
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([# 创建多个Transformer块
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.time_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()# 修正初始化权重

    def fix_init_weight(self):
        def rescale(param, layer_id):# 修正权重，使用 layer_id 的平方根进行缩放
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):# 对每个Transformer块的注意力权重和MLP权重进行缩放
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        # from IPython import embed; embed()
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)
        
        x = self.norm(x)
        return x
        # if self.fc_norm is not None:
        #     if return_all_tokens:
        #         return self.fc_norm(x)
        #     t = x[:, 1:, :]
        #     if return_patch_tokens:
        #         return self.fc_norm(t)
        #     else:
        #         return self.fc_norm(t.mean(1))#走的这里
        # else:
        #     if return_all_tokens:
        #         return x
        #     elif return_patch_tokens:
        #         return x[:, 1:]
        #     else:
        #         return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        '''
        # print("modeling_finetune.436")
        # print("sample.shape2:",sst.shape)
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)#[64,200]
        # from IPython import embed; embed()
        return x
    
    
    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)# 如果没有传入 Mask 的布尔矩阵，则默认所有位置都未被 Mask
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)# 使用 Patch 嵌入层提取初步特征
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 复制 [CLS] Token 和 Mask Token
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked EEG tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)# 拼接 [CLS] Token
        if self.pos_embed is not None:
            x = x + self.pos_embed# 添加位置编码
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)# 除最后一层外，逐层通过 Transformer Block
            else:
                # with torch.cuda.amp.autocast(enabled=False):
                x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)# 最后一层计算 QKV 输出

        if split_out_as_qkv:# 如果需要分别返回 Q, K, V
            x = self.norm(x)
            x = self.lm_head(x) # [b, n+1, 3*c]
            q, k, v = x.chunk(3, dim=-1) # [b, n+1, c]
            b, n, c =q.shape
            q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)# 重塑 Q, K, V 为 [B, num_heads, seq_len, head_dim] 
            k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            return x, q, k, v
        else:
            x = self.norm(x)
            x = x[:, 1:]
            x = self.lm_head(x[bool_masked_pos])

            q, k, v = qkv[0], qkv[1], qkv[2]

        return x, q, k, v

    def get_last_selfattention(self, x):
        x = self.patch_embed(x)# 使用 Patch 嵌入层提取初步特征
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)# 复制 [CLS] Token
        x = torch.cat((cls_tokens, x), dim=1) # 拼接 [CLS] Token
        if self.pos_embed is not None:
            x = x + self.pos_embed # 添加位置编码
        x = self.pos_drop(x) # 应用 Dropout
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias) # 除最后一层外，逐层通过 Transformer Block
            else:
                # return attention of the last block
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)# 返回最后一层的注意力权重
            

class GT_NeuralTransformerForMEM(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.student = GT_NeuralTransformerForMaskedEEGModeling(EEG_size, patch_size, in_chans, out_chans, vocab_size, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, attn_head_dim,
                 use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std)
        
        self.lm_head = nn.Linear(embed_dim, vocab_size)# 语言建模头
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.ReLU()
        )

        trunc_normal_(self.lm_head.weight, std=init_std)# 初始化语言建模头权重
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'student.cls_token', 'student.pos_embed', 'student.time_embed'}
    
    def forward(self, x, input_chans=None, bool_masked_pos=None):
        # from IPython import embed; embed()
        x = self.student(x, input_chans, return_all_patch_tokens=True)
        # from IPython import embed; embed()
        x=x[:,1:,]
        # from IPython import embed; embed()
        predict = self.lm_head(x[bool_masked_pos])
        predict_sym=self.lm_head(x[~bool_masked_pos])
        
        return predict,predict_sym


@register_model
def gt_labram_base_patch200_1600_8k_vocab(pretrained=False, **kwargs): #5M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']# 提取词汇表大小
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = GT_NeuralTransformerForMEM(# 初始化 NeuralTransformerForMEM 模型
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def gt_labram_large_patch200_1600_8k_vocab(pretrained=False, **kwargs): #50M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = GT_NeuralTransformerForMEM(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def gt_labram_huge_patch200_1600_8k_vocab(pretrained=False, **kwargs): #380M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = GT_NeuralTransformerForMEM(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=32,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model





















class MambaNeuralTransformerForMaskedEEGModeling(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = TemporalConv(out_chans=out_chans)
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))# 分类标记 (CLS token)，初始化为全零
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))# Mask标记，用于替换遮掩的输入

        if use_abs_pos_emb:# 如果使用绝对位置编码
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))# 位置编码 (含CLS token)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)# 时间嵌入，支持时间维度的信息
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([# 创建多个Transformer块
            MambaBlock(
                c_in=embed_dim,m_layers=2,d_model=32,d_ff=256,dropout=0.,act='gelu',d_state=16,d_conv=4,m_patch_len=16,m_stride=8,#最后这个胡乱设置的
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.time_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        # self.fix_init_weight()# 修正初始化权重

    # def fix_init_weight(self):
    #     def rescale(param, layer_id):# 修正权重，使用 layer_id 的平方根进行缩放
    #         param.div_(math.sqrt(2.0 * layer_id))

    #     for layer_id, layer in enumerate(self.blocks):# 对每个Transformer块的注意力权重和MLP权重进行缩放
    #         rescale(layer.attn.proj.weight.data, layer_id + 1)
    #         rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, input_chans, bool_masked_pos): # 提取特征的主函数
        
        input_chans=[0, 28, 29, 30, 31, 32, 33, 34, 39, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 56, 1, 2, 3, 5, 7, 9, 11, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 35, 38, 46, 37, 47, 49, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 71, 73, 75, 77, 79, 81, 82, 83, 87]
        
        # qprint(input_chans,'input_chans')
        batch_size, c, time_window, _ = x.size()
        # qprint(x.shape,'x.shape')
        x = self.patch_embed(x)
        # qprint(x.shape,'x.shape')
        batch_size, seq_len, _ = x.size()
        # qprint(x.shape,'x.shape')

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)   # 复制CLS标记
        mask_token = self.mask_token.expand(batch_size, seq_len, -1) # 复制Mask标记

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token) # 将Mask位转换为权重
        x = x * (1 - w) + mask_token * w # 替换被Mask掉的Patch

        x = torch.cat((cls_tokens, x), dim=1)
        # qprint(x.shape,'x.shape')
        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        # qprint(pos_embed_used.shape,'pos_embed_used.shape')
        if self.pos_embed is not None: # 处理位置编码
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            # qprint(pos_embed.shape,'pos_embed.shape')
            x = x + pos_embed
        if self.time_embed is not None:# 处理时间编码
            time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        #这之后的x的shape应该是什么样的
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        # x=x.permute(0,2,1)
        for blk in self.blocks:#B,seqlen,dim
            # qprint(x.shape,'front:x.shape')
            # qprint(x.shape,'x.shape')
            x = blk(x)
            # x=x.permute(0,2,3,1)#[B x nvars x m_patch_num x d_model]
            # x=x.reshape(x.shape[0],-1,x.shape[3])
            # qprint(x.shape,'back:x.shape')
            # x=x.permute(0,2,1)

            # x=x.permute(0,2,1)

        return self.norm(x)# 输出归一化后的特征

    def forward(self, x, input_chans=None, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False, return_all_patch_tokens=False):
        # qprint(x.shape,'x.shape')
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)# 如果没有传入 Mask 的布尔矩阵，则默认所有位置都未被 Mask
        x = self.forward_features(x, input_chans=input_chans, bool_masked_pos=bool_masked_pos)# 使用 `forward_features` 提取特征
        if return_all_patch_tokens:# 如果需要返回所有 Patch 特征，直接返回
            return x
        x = x[:, 1:]# 丢弃第一个位置的 [CLS] Token
        qprint(x.shape,'476.x.shape')
        if return_patch_tokens:# 如果只需要 Patch Tokens 的特征，返回结果
            return x
        if return_all_tokens:# 如果需要对所有 Tokens 应用输出层，直接返回语言建模头的输出
            return self.lm_head(x)
        else:
            tmp=self.lm_head(x[bool_masked_pos])
            qprint(tmp.shape,'tmp.shape')
            # return the masked tokens
            return self.lm_head(x[bool_masked_pos])# 返回被 Mask 的位置对应的输出
    
    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)# 如果没有传入 Mask 的布尔矩阵，则默认所有位置都未被 Mask
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)# 使用 Patch 嵌入层提取初步特征
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 复制 [CLS] Token 和 Mask Token
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked EEG tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)# 拼接 [CLS] Token
        if self.pos_embed is not None:
            x = x + self.pos_embed# 添加位置编码
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)# 除最后一层外，逐层通过 Transformer Block
            else:
                # with torch.cuda.amp.autocast(enabled=False):
                x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)# 最后一层计算 QKV 输出

        if split_out_as_qkv:# 如果需要分别返回 Q, K, V
            x = self.norm(x)
            x = self.lm_head(x) # [b, n+1, 3*c]
            q, k, v = x.chunk(3, dim=-1) # [b, n+1, c]
            b, n, c =q.shape
            q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)# 重塑 Q, K, V 为 [B, num_heads, seq_len, head_dim] 
            k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            return x, q, k, v
        else:
            x = self.norm(x)
            x = x[:, 1:]
            x = self.lm_head(x[bool_masked_pos])

            q, k, v = qkv[0], qkv[1], qkv[2]

        return x, q, k, v

    def get_last_selfattention(self, x):
        x = self.patch_embed(x)# 使用 Patch 嵌入层提取初步特征
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)# 复制 [CLS] Token
        x = torch.cat((cls_tokens, x), dim=1) # 拼接 [CLS] Token
        if self.pos_embed is not None:
            x = x + self.pos_embed # 添加位置编码
        x = self.pos_drop(x) # 应用 Dropout
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias) # 除最后一层外，逐层通过 Transformer Block
            else:
                # return attention of the last block
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)# 返回最后一层的注意力权重
            

class MambaNeuralTransformerForMEM(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.student = MambaNeuralTransformerForMaskedEEGModeling(EEG_size, patch_size, in_chans, out_chans, vocab_size, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, attn_head_dim,
                 use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std)
        
        self.lm_head = nn.Linear(embed_dim, vocab_size)# 语言建模头
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.ReLU()
        )

        trunc_normal_(self.lm_head.weight, std=init_std)# 初始化语言建模头权重
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'student.cls_token', 'student.pos_embed', 'student.time_embed'}
    
    def forward(self, x, input_chans=None, bool_masked_pos=None):
        # qprint(x.shape,'571.x.shape')
        x_masked = self.student(x, input_chans, bool_masked_pos, return_all_patch_tokens=True)#, input_chans, bool_masked_pos
        # qprint(x_masked.shape,'573.x_masked.shape')
        x_masked_no_cls = x_masked[:, 1:]# 丢弃 [CLS] Token
        # qprint(bool_masked_pos.shape,'bool_masked_pos.shape')
        # qprint(x_masked_no_cls.shape,'x_masked_no_cls.shape')
        x_rec = self.lm_head(x_masked_no_cls[bool_masked_pos])# 重建被 Mask 的位置
        # qprint('1','1')
        #symetric 对称 Mask：处理 Mask 和未 Mask 的情况
        x_masked_sym = self.student(x, input_chans, ~bool_masked_pos, return_all_patch_tokens=True)
        # qprint('2','2')
        x_masked_no_cls_sym = x_masked_sym[:, 1:]
        # qprint('3','3')
        x_rec_sym = self.lm_head(x_masked_no_cls_sym[~bool_masked_pos])
        # qprint('1','1')

        return x_rec, x_rec_sym # 返回两种重建结果

@register_model
def mamba_labram_base_patch200_1600_8k_vocab(pretrained=False, **kwargs): #5M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']# 提取词汇表大小
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    # qprint('heoo','hello')
    model = MambaNeuralTransformerForMEM(# 初始化 NeuralTransformerForMEM 模型
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

