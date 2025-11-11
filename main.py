# main.py:
#   for now lets just build and run sam2 across a couple gpus (w/ TPA)


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import comm


# placeholder config class
# TODO: replace with SAM2 config
#   - possibly use AutoConfig because we want this to be fully
#       generalizable
#   - as of now, input should be a config, output should be a list
#       of layers and dimensions used to be the model
class Cfg:
    d_model = 1024
    n_heads = 32
    seq_len = 64
    device = "cuda"
    dtype = torch.float16
    
class ShardedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        
        #
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = comm.rank()
        self.world_size() = comm.world_size()

        # compute local out_features slice (equal split assumed):
        assert out_features % self.world_size == 0, "out_features must be divisible by world_size"
        self.local_out = out_features // self.world_size
        
        # allocate local weight on device
        self.weight = nn.Parameter(torch.empty((self.local_out, in_features), device=Cfg.device, dtype=Cfg.dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.local_out, device=Cfg.device, dtype=Cfg.dtype))
        else:
            self.register_parameter("bias", None)

        # init
        nn.init.normal_(self.weight, mean=0.0, std=0.02)


    def forward(self, x):
        # x: (batch, seq, in_features)
        # local matmul -> (batch, seq, local_out)
        out = torch.einsum("bsi, oi->bso", x, self.weight) # NOTE: equivalent to "x @ weight.T"
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        return out
    
class TPAttention(nn.Module):
    """
    tensor parallel attention, split heads across ranks
    (each rank computers its head outputs and they are all_gathered)

    - NOTE: assumes number of heads can be split evenly on gpus (world_size)
    """

    def __init__(self, d_model, n_heads):

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.world_size = comm.world_size()
        self.rank = comm.rank()
        assert n_heads % self.world_size == 0

        # setup split params
        self.local_heads = n_heads // self.world_size
        self.head_dim = d_model // n_heads


        self.qkv = ShardedLinear(d_model, 3 * d_model, bias=True)

        if comm.world_size() == 1:
            self.out_proj = nn.Linear(d_model, d_model).to(Cfg.device).to(Cfg.dtype)
        else:
            # more than 1 gpu so parallelize
            # for now just keep the same for correctness checks
            # TODO: gather all head outputs, run output projection only on rank 0 in sharded form
            self.out_proj = nn.Linear(d_model, d_model).to(Cfg.device).to(Cfg.dtype)

    def forward(self, x, mask: Optional[torch.Tensor] = None):

        # x: (batch, seq, d_model), we only need b and s here
        batch, seq, _ = x.shape

        # local qkv: (batch, seq, local_out) where local_out = (3*d_model)/world_size
        # NOTE: this is where ShardedLinear is called
        #   essentially a learned matrix transformation
        qkv_local = self.qkv(x)
        
        # reshape local_out into (batch, seq, 3, local_heads, head_dim)
        # NOTE: 3 hardcoded here for q,k,v
        local_out = qkv_local.view(batch, seq, 3, self.local_heads, self.head_dim)
        
        # get everything on dimension but select q k v respectively for local tensors
        q_local = local_out[:, :, 0] # batch seq local_heads head_dim
        k_local = local_out[:, :, 1]
        v_local = local_out[:, :, 2]

        # compute attention per-local-head
        # (permute) reorder to (batch, heads, seq, head_dim) 
        # (reshape) to combine batch and heads into b_h
        b_h = batch * self.local_heads
        q_ = q_local.permute(0,2,1,3).reshape(b_h, seq, self.head_dim)
        k_ = k_local.permute(0,2,1,3).reshape(b_h, seq, self.head_dim)
        v_ = v_local.permute(0,2,1,3).reshape(b_h, seq, self.head_dim)

        # scaled dot-product attention (causal mask can be applied externally)
        # (bmm) batch matmult -> mult pairs of 2d matrices across batch dimension
        # (transpose) swaps last two axes for propery mult
        # results in (b_h, seq, seq) NOTE: these are attn SCORES between tokens
        # lastly divide by sqrt(head_dim) for stability
        scores = torch.bmm(q_, k_.transpose(1,2)) / math.sqrt(self.head_dim)

        # if mask exists, must duplicate across head dimension so each head gets one
        # (masked_fill) sets positions in mask that are 0 to negative infinity
        #   which ignores them in the softmax
        if mask is not None:
            # mask shape should be (batch, 1, seq, seq) or broadcastable
            # expand mask across local_heads
            mask_exp = mask.repeat_interleave(self.local_heads, dim=0)
            scores = scores.masked_fill(mask_exp == 0, float("-inf"))

        # (softmax) turns scores into probabilities along last axis
        # (bmm) computes the weighted sum of values v_ according to attn weights
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, v_)  # (b_h, seq, head_dim)

        # reconstruct original shape (undo flattening)
        # must become: (batch, seq, all_head_features).
        out = out.view(batch, self.local_heads, seq, self.head_dim)
        out = out = out.permute(0,2,1,3).reshape(batch, seq, self.local_heads * self.head_dim)

        # each gpu computes a slice
        # custom all gather collects allk slices so each gpu has final result
        gathered = comm.all_gather_tensor_on_gpu(out) # NOTE: concat on last dim (heads)
        
        # (out_proj) second linear layer
        # full should be (batch, seq, d_model)
        full = self.out_proj(gathered)
        return full

# Minimal condensed SAM2-like block using TPAttention
class MinimalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = TPAttention(cfg.d_model, cfg.n_heads).to(Cfg.device).to(Cfg.dtype)
        # small MLP (kept non-sharded for simplicity)
        self.mlp_fc1 = nn.Linear(cfg.d_model, cfg.d_model * 4).to(Cfg.device).to(Cfg.dtype)
        self.mlp_fc2 = nn.Linear(cfg.d_model * 4, cfg.d_model).to(Cfg.device).to(Cfg.dtype)
        self.ln1 = nn.LayerNorm(cfg.d_model).to(Cfg.device).to(torch.float32)  # layernorm in fp32
        self.ln2 = nn.LayerNorm(cfg.d_model).to(Cfg.device).to(torch.float32)

    def forward(self, x, mask=None):
        # keep layernorm in fp32 for stability
        x_fp32 = x.to(torch.float32)
        attn_out = self.attn(self.ln1(x_fp32).to(Cfg.dtype), mask)
        x = x + attn_out
        x_fp32 = x.to(torch.float32)
        mlp_out = self.mlp_fc2(F.gelu(self.mlp_fc1(self.ln2(x_fp32).to(Cfg.dtype))))
        x = x + mlp_out
        return x
