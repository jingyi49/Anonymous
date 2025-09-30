import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from utils import init_weights, get_padding
import numpy as np
from quantize import ResidualVectorQuantize

from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Transformer
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat, pack, unpack



class Encoder(torch.nn.Module):
    def __init__(self, h):
        super(Encoder, self).__init__()
        
        self.h = h
        self.temporal_patch_size = h.temporal_patch_size
        self.frequency_patch_size = h.frequency_patch_size
        self.patch_embed = h.patch_embed
        self.defer_temporal_pool = h.defer_temporal_pool
        self.defer_frequency_pool = h.defer_frequency_pool
        self.dim = h.latent_dim
        self.wave_channel = h.wave_channel
        self.window_size = h.window_size
        self.temporal_depth = h.temporal_depth
        self.frequency_depth = h.frequency_depth
        self.dim_head = h.dim_head
        self.heads: int = h.heads
        self.attn_dropout: float = h.attn_dropout
        self.ff_dropout: float = h.ff_dropout
        self.ff_mult: float = h.ff_mult
        self.causal_in_peg: bool = h.causal_in_peg
        self.initialize: bool = h.initialize

        self.quantizer = ResidualVectorQuantize(
            input_dim=h.latent_dim,
            codebook_dim=h.codebook_dim,
            n_codebooks=4,
            codebook_size=1024,
            quantizer_dropout=False
        )
        
        if self.patch_embed == 'linear':
            if h.defer_temporal_pool:
                h.temporal_patch_size //= 2
                self.temporal_patch_size = h.temporal_patch_size
                #只在最后一维
                self.temporal_pool = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
                #kernel_size=2表示在时间维度每 2 个点做平均，stride=2 → 下采样一半。输入[B,C,T]输出[B,C,T//2]
            else:
                self.temporal_pool = nn.Identity()
            
            if h.defer_frequency_pool:
                h.frequency_patch_size //= 2
                self.frequency_patch_size = h.frequency_patch_size
                #只在倒数第二维
                self.frequency_pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
            else:
                self.frequency_pool = nn.Identity()
            
            self.to_patch_emb = nn.Sequential(
                # 将 [B, C, F, T] 切 patch: F = freq, T = time
                Rearrange('b c (f pf) (t pt) -> b f t (c pf pt)',
                        pt=h.temporal_patch_size, pf=h.frequency_patch_size),
                nn.LayerNorm(h.wave_channel * h.frequency_patch_size * h.temporal_patch_size),
                nn.Linear(h.wave_channel * h.frequency_patch_size * h.temporal_patch_size, h.latent_dim),
                nn.LayerNorm(h.latent_dim)
                )

        elif patch_embed == 'cnn':
            self.to_patch_emb = nn.Sequential(
                nn.Conv2d(
                    in_channels=h.wave_channel,
                    out_channels=h.latent_dim,
                    kernel_size=(h.frequency_patch_size, h.temporal_patch_size),
                    stride=(h.frequency_patch_size, h.temporal_patch_size)
                ),
                Normalize(h.latent_dim, norm_type),
                Rearrange('b c f t -> b f t c')
            )
            self.temporal_pool, self.spatial_pool = nn.Identity(), nn.Identity()
        
        else:
            raise NotImplementedError
        
        transformer_kwargs = dict(
            dim=self.dim,
            dim_head=self.dim_head,
            heads=self.heads,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            peg=True,
            peg_causal=self.causal_in_peg,
            ff_mult=self.ff_mult
        )
        
        self.enc_frequency_transformer = Transformer(depth=self.frequency_depth, block="ttww", window_size=self.window_size, spatial_pos="rel", **transformer_kwargs)
        
        self.enc_temporal_transformer = Transformer(
            depth=self.temporal_depth, block='t' * self.temporal_depth, **transformer_kwargs)

        if self.initialize:
            self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @property
    def patch_width(self):
        return self.waveform_size[0] // self.frequency_patch_size, self.waveform_size[1] // self.temporal_patch_size

    def encode(
        self,
        tokens
    ):  
        """
        tokens: [B, F_patches, T_patches, D]
        B: batch size
        F_patches: 频率方向 patch 数
        T_patches: 时间方向 patch 数
        D: embedding dim
        b, f, t, d = tokens.shape
        """
        b = tokens.shape[0]  # batch size
        spectral_shape= tuple(tokens.shape[:-1]) #the last dim is embedding dim
        tokens = rearrange(tokens, 'b f t d -> (b t) f d')
        tokens = self.enc_frequency_transformer(tokens, waveform_shape=spectral_shape, is_spatial=True)

        new_f = tokens.shape[1]
        tokens = rearrange(tokens, '(b t) f d -> b f t d', b=b, f=new_f)
        spectral_shape = tuple(tokens.shape[:-1])
        tokens = rearrange(tokens, 'b f t d -> (b f) t d', b=b, f=new_f)
        tokens = self.enc_temporal_transformer(tokens, waveform_shape=spectral_shape, is_spatial=False)

        # codebook expects:  [b, c, f, t] 
        tokens = rearrange(tokens, '(b f) t d -> b d f t', b=b, f=new_f)
        tokens = self.frequency_pool(tokens)
        return tokens

    def forward(self, spectral, mask=None):
        """
        spectral: (B, C, F, T)  e.g., B=批大小, C=通道数(1=单声道), F=频率长度, T=时间点
        mask:  (可选) (B, N) token mask
        """
        # patch embedding: Conv1d 或 Linear
        tokens = self.to_patch_emb(spectral)   # (B, f, t, dim)

        if mask is not None:
            assert mask.shape[1] == tokens.shape[1], \
                f"mask shape {mask.shape} does not match tokens {tokens.shape}"
        
        latent = self.encode(tokens) #[b, c, f, t] 变为 [b,c,(ft)]
        b, c, f, t = latent.shape
        latent = rearrange(latent, 'b c f t -> b c (f t)')
        latent,_,_,commitment_loss,codebook_loss = self.quantizer(latent)
        latent = rearrange(latent, 'b c (f t) -> b c f t', f=f, t=t)
        return latent,commitment_loss,codebook_loss  # (B, C, F_pooled, T_pooled)  # codebook 输出或 encoder 输出


class Decoder(torch.nn.Module):
    def __init__(self, h):
        super(Decoder, self).__init__()
        self.temporal_patch_size = h.temporal_patch_size
        self.frequency_patch_size = h.frequency_patch_size
        self.patch_embed = h.patch_embed
        self.defer_temporal_pool = h.defer_temporal_pool
        self.defer_frequency_pool = h.defer_frequency_pool
        self.dim = h.latent_dim
        self.wave_channel = h.wave_channel
        self.window_size = h.window_size
        self.temporal_depth = h.temporal_depth
        self.frequency_depth = h.frequency_depth
        self.dim_head = h.dim_head
        self.heads: int = h.heads
        self.attn_dropout: float = h.attn_dropout
        self.ff_dropout: float = h.ff_dropout
        self.ff_mult: float = h.ff_mult
        self.causal_in_peg: bool = h.causal_in_peg
        self.initialize: bool = h.initialize
        
        transformer_kwargs = dict(
            dim=self.dim,
            dim_head=self.dim_head,
            heads=self.heads,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            peg=True,
            peg_causal=self.causal_in_peg,
            ff_mult=self.ff_mult
        )

        self.dec_temporal_transformer = Transformer(
            depth=self.temporal_depth, block='t' * self.temporal_depth, **transformer_kwargs)
        self.dec_freq_transformer = Transformer(depth=self.frequency_depth, block="wwtt", window_size=self.window_size, spatial_pos="rel", **transformer_kwargs)
        if self.patch_embed == "linear":
            if self.defer_temporal_pool:
                h.temporal_patch_size //= 2
                self.temporal_patch_size = h.temporal_patch_size
                self.temporal_up = nn.Upsample(scale_factor=(1, 2), mode="nearest")
            else:
                self.temporal_up = nn.Identity()

            if self.defer_frequency_pool:
                frequency_patch_size //= 2
                self.frequency_patch_size = frequency_patch_size
                self.frequency_up = nn.Upsample(scale_factor=(2, 1), mode="nearest")  
            else:
                self.frequency_up = nn.Identity()

            self.from_patch_emb = nn.Sequential(
                        nn.LayerNorm(self.dim),                                 # 先归一化
                        nn.Linear(self.dim, self.wave_channel * self.temporal_patch_size*self.frequency_patch_size),# 还原回 patch
                        Rearrange('b f t (c pf pt) -> b c (f pf) (t pt)', pt=self.temporal_patch_size,pf=self.frequency_patch_size))
        
        elif self.patch_embed == "cnn":
            
            self.from_patch_emb = nn.Sequential(
                Rearrange('b f t c -> b c f t'),                     # 转回 Conv1d 的输入格式
                nn.ConvTranspose2d(
                    in_channels=self.dim,
                    out_channels=self.wave_channel,
                    kernel_size=(self.frequency_patch_size, self.temporal_patch_size),           # 还原到原 patch 大小
                    stride=(self.frequency_patch_size, self.temporal_patch_size)
                ),
                Normalize(self.dim, norm_type),
            )
            
            self.temporal_up = nn.Identity()
            self.frequency_up = nn.Identity()
        
        else:
            raise NotImplementedError

        if self.initialize:
            self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @property
    def patch_width(self):
        return self.waveform_size[0] // self.frequency_patch_size, self.waveform_size[1] // self.temporal_patch_size


    def decode(self, tokens):
        """
        tokens: (B, C, F, T)  # codebook 或 encoder 输出
        B: batch size
        C: embedding dim
        F: 频率长度
        T: 时间长度
        """
        b = tokens.shape[0]
        b, f, t, c = tokens.shape #([13, 10, 32, 320])
        #decode-temperal
        tokens = rearrange(tokens, 'b f t c -> (b f) t c')
        tokens = self.dec_temporal_transformer(tokens, waveform_shape=(b,f,t), is_spatial=False)
        

        #decode-frequency
        new_t = tokens.shape[1]
        tokens = rearrange(tokens, '(b f) t c -> (b t) f c', b=b, f=f, t=new_t)
        tokens =  self.dec_freq_transformer(tokens, waveform_shape=(b,f,t), is_spatial=True)
        
        tokens = rearrange(tokens, '(b t) f c -> b f t c', b=b)
        if self.patch_embed == 'linear':
            tokens = self.from_patch_emb(tokens)  # nn.Linear 逆投影
        elif self.patch_embed == 'cnn':
            tokens = self.from_patch_emb(tokens)  
        
        return tokens  # B, C, T
   
    
    def forward(self, tokens, mask=None):
        """
        tokens: (B, C, T)  # codebook 输出或 encoder 输出
        mask:  (可选) (B, N) token mask
        """
        # 1. temporal upsampling (inverse of encoder pooling)
        tokens = self.temporal_up(tokens)  # B, C, T_up
        tokens = self.frequency_up(tokens)  # B, C, F_up, T_up

        # 2. rearrange 到 [B, N, D] 供 Transformer decode
        tokens = rearrange(tokens, 'b c f t -> b f t c')

        if mask is not None:
            assert mask.shape[1] == tokens.shape[1], \
                f"mask shape {mask.shape} does not match tokens {tokens.shape}"

        recon = self.decode(tokens)
        #if self.wave_channel==1:
        #    recon = recon.squeeze(1)
        return recon


