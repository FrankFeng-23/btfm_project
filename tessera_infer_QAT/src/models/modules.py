# src/models/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, 1)
    def forward(self, x):
        # More explicit operations
        w = torch.softmax(self.query(x), dim=1)
        weighted = w * x
        return weighted.sum(dim=1)  # More explicit sum
 
class ConvTemporalPooling(nn.Module):
    """使用1D卷积捕获时序依赖的池化方法"""
    def __init__(self, input_dim, kernel_sizes=[3, 5, 7]):
        super().__init__()
        num_kernels = len(kernel_sizes)
        # Calculate dimensions more carefully to avoid mismatch
        conv_out_dim_per_kernel = input_dim // num_kernels
        # Total dimension after concatenation
        total_conv_out_dim = conv_out_dim_per_kernel * num_kernels
        
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, conv_out_dim_per_kernel, 
                     kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Use the actual concatenated dimension for query
        self.query = nn.Linear(total_conv_out_dim, 1)
        
        # Add a projection layer to map back to original dimension if needed
        self.proj_back = None
        if total_conv_out_dim != input_dim:
            self.proj_back = nn.Linear(total_conv_out_dim, input_dim)
        
    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        if T == 0:
            return torch.zeros(B, D, device=x.device)
        elif T == 1:
            return x.squeeze(1)
            
        # 转换为Conv1d格式: (B, D, T)
        x_conv = x.transpose(1, 2)
        
        # 多尺度卷积提取时序特征
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x_conv))
        
        # 合并多尺度特征
        x_multi = torch.cat(conv_outs, dim=1)  # (B, total_conv_out_dim, T)
        x_multi = x_multi.transpose(1, 2)  # (B, T, total_conv_out_dim)
        
        # 计算注意力权重
        w = torch.softmax(self.query(x_multi), dim=1)  # (B, T, 1)
        
        # If dimensions don't match, project the multi-scale features back
        if self.proj_back is not None:
            x_multi = self.proj_back(x_multi)  # (B, T, D)
            # Apply attention to projected features
            return (w * x_multi).sum(dim=1)  # (B, D)
        else:
            # Apply attention to original input
            return (w * x).sum(dim=1)  # (B, D)

   
class TemporalAwarePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, 1)
        self.temporal_context = nn.GRU(input_dim, input_dim, batch_first=True)
        
    # def forward(self, x):
    #     # 先通过RNN捕获时序上下文
    #     x_context, _ = self.temporal_context(x)
    #     # 再计算注意力权重
    #     w = torch.softmax(self.query(x_context), dim=1)
    #     return (w * x).sum(dim=1)
    def forward(self, x):
        # Add safety check for sequence length
        B, T, D = x.shape
        if T == 0:
            # Handle empty sequence case - return zeros
            return torch.zeros(B, D, device=x.device)
        elif T == 1:
            # Handle single element sequence - skip GRU
            return x.squeeze(1)
        else:
            # Normal case with multiple timesteps
            try:
                # Process through RNN to capture temporal context
                x_context, _ = self.temporal_context(x)
                # Calculate attention weights
                w = torch.softmax(self.query(x_context), dim=1)
                return (w * x).sum(dim=1)
            except Exception as e:
                # Fallback to mean pooling if RNN fails
                print(f"RNN failed with error: {e}. Falling back to mean pooling, current x shape is {x.shape}, values are {x}")
                return torch.mean(x, dim=1)


class CustomGRUCell(nn.Module):
    """自定义 GRU Cell 实现，仅使用基本 torch 操作"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入到门的权重
        self.W_ir = nn.Linear(input_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        
        # 隐藏状态到门的权重
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 偏置
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        # 使用 Xavier 初始化
        for name, param in self.named_parameters():
            if 'weight' in name or name.startswith('W_'):
                nn.init.xavier_uniform_(param)
                
    def forward(self, x_t, h_prev):
        """
        前向传播单个时间步
        x_t: (batch_size, input_size)
        h_prev: (batch_size, hidden_size)
        """
        # 重置门
        r_t = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_prev) + self.b_r)
        
        # 更新门
        z_t = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_prev) + self.b_z)
        
        # 候选隐藏状态
        h_tilde = torch.tanh(self.W_ih(x_t) + self.W_hh(r_t * h_prev) + self.b_h)
        
        # 新的隐藏状态
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t


class CustomGRU(nn.Module):
    """自定义 GRU 层实现"""
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        self.gru_cell = CustomGRUCell(input_size, hidden_size)
        
    def forward(self, x, h_0=None):
        """
        x: (batch_size, seq_len, input_size) if batch_first=True
        h_0: 初始隐藏状态 (batch_size, hidden_size)
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape
            x = x.transpose(0, 1)
        
        # 初始化隐藏状态
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, 
                            device=x.device, dtype=x.dtype)
        
        # 存储所有时间步的输出
        outputs = []
        h_t = h_0
        
        # 循环处理每个时间步
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = self.gru_cell(x_t, h_t)
            outputs.append(h_t)
        
        # 堆叠所有输出
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)
            
        return outputs, h_t


class CustomTemporalAwarePooling(nn.Module):
    """使用自定义 GRU 实现的时序感知池化"""
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # 使用自定义 GRU 替代 nn.GRU
        self.temporal_context = CustomGRU(input_dim, input_dim, batch_first=True)
        
        # 注意力查询层
        self.query = nn.Linear(input_dim, 1)
        
        # 可选：添加层归一化以提高稳定性
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        B, T, D = x.shape
        
        # 处理边界情况
        if T == 0:
            return torch.zeros(B, D, device=x.device, dtype=x.dtype)
        elif T == 1:
            return x.squeeze(1)
        
        try:
            # 通过自定义 GRU 捕获时序上下文
            x_context, _ = self.temporal_context(x)
            
            # 可选：应用层归一化
            x_context = self.layer_norm(x_context)
            
            # 计算注意力权重
            attn_scores = self.query(x_context)  # (B, T, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            
            # 应用注意力权重
            weighted_x = attn_weights * x  # (B, T, D)
            pooled = weighted_x.sum(dim=1)  # (B, D)
            
            return pooled
            
        except Exception as e:
            # 备用方案：如果出现任何错误，使用平均池化
            print(f"Custom GRU failed with error: {e}. Falling back to mean pooling.")
            print(f"Input shape: {x.shape}")
            return torch.mean(x, dim=1)

class TemporalPositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
       
    def forward(self, doy):
        # doy: [B, T] tensor containing DOY values (0-365)
        position = doy.unsqueeze(-1).float()  # Ensure float type
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float) * -(math.log(10000.0) / self.d_model))
        div_term = div_term.to(doy.device)
       
        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    

class TransformerEncoder(nn.Module):
    def __init__(self, band_num, latent_dim, nhead=8, num_encoder_layers=4,
                dim_feedforward=512, dropout=0.1, max_seq_len=20):
        super().__init__()
        # Total input dimension: bands
        input_dim = band_num
        
        # Embedding to increase dimension
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, latent_dim*4),
            nn.ReLU(),
            nn.Linear(latent_dim*4, latent_dim*4)
        )
        
        # Temporal Encoder for DOY as position encoding
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim*4)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim*4,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
       
        # Temporal Aware Pooling
        # self.attn_pool = TemporalAwarePooling(latent_dim*4)
        
        # Attention Pooling
        # self.attn_pool = AttentionPooling(latent_dim*4)
        
        # CNN-based Temporal Pooling
        # self.attn_pool = ConvTemporalPooling(latent_dim*4, kernel_sizes=[3, 5, 7])
        
        # custom GRU-based Temporal Pooling
        self.attn_pool = CustomTemporalAwarePooling(latent_dim*4)
   
    def forward(self, x):
        # x: (B, seq_len, 10 bands + 1 doy)
        # Split bands and doy
        bands = x[:, :, :-1]  # All columns except last one
        doy = x[:, :, -1]     # Last column is DOY
        # Embedding of bands
        bands_embedded = self.embedding(bands)  # (B, seq_len, latent_dim*4)
        temporal_encoding = self.temporal_encoder(doy)
        # Add temporal encoding to embedded bands (instead of random positional encoding)
        x = bands_embedded + temporal_encoding
        x = self.transformer_encoder(x)
        x = self.attn_pool(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(hidden_dim, output_dim),
        # )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            
            # nn.Linear(hidden_dim, hidden_dim//2),
            # nn.BatchNorm1d(hidden_dim//2),
            # nn.ReLU(inplace=False),
            # nn.Linear(hidden_dim//2, hidden_dim//4),
            # nn.BatchNorm1d(hidden_dim//4),
            # nn.ReLU(inplace=False),
            # nn.Linear(hidden_dim//4, hidden_dim//8),
            # nn.BatchNorm1d(hidden_dim//8),
            # nn.ReLU(inplace=False),
            # nn.Linear(hidden_dim//8, hidden_dim//4),
            # nn.BatchNorm1d(hidden_dim//4),
            # nn.ReLU(inplace=False),
            # nn.Linear(hidden_dim//4, hidden_dim//2),
            # nn.BatchNorm1d(hidden_dim//2),
            # nn.ReLU(inplace=False),
            # nn.Linear(hidden_dim//2, hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)
 


