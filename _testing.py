import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from chatgpt, prompt for MHA "basic template"

class MHA_1(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # projection layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # final linear
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        # 1. project
        Q = self.W_q(x)  # (B, S, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. reshape into heads
        def reshape_heads(t):
            return t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            # → (B, heads, S, head_dim)

        Q, K, V = reshape_heads(Q), reshape_heads(K), reshape_heads(V)

        # 3. attention scores
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        # scores: (B, heads, S, S)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = scores.softmax(dim=-1)
        out = attn @ V   # (B, heads, S, head_dim)

        # 4. combine heads
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        # 5. final projection
        return self.W_o(out)




# from https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    # (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len) --> (batch, heads, seq_len, seq_len)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim) --> (batch, heads, seq_len, head_dim)
    values = torch.matmul(attention, v)
    return values, attention

class MHA_2(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim      # Input embedding size
        self.d_model = d_model          # Model embedding size (output of self-attention)
        self.num_heads = num_heads      # Number of parallel attention heads
        self.head_dim = d_model // num_heads  # Dimensionality per head

        # For efficiency, compute Q, K, V for all heads at once with a single linear layer
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        # Final projection, combines all heads' outputs
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        print(f"x.size(): {x.size()}")  # Input shape

        # Step 1: Project x into concatenated q, k, v for ALL heads at once
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")  # Shape: (batch, seq_len, 3 * d_model)

        # Step 2: reshape into (batch, seq_len, num_heads, 3 * head_dim)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")

        # Step 3: Rearrange to (batch, num_heads, seq_len, 3 * head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")

        # Step 4: Split the last dimension into q, k, v (each get last dimension of head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (batch, num_heads, seq_len, head_dim)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}")

        # Step 5: Apply scaled dot product attention to get outputs (contextualized values) and attention weights
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"values.size(): {values.size()}, attention.size: {attention.size()}")

        # Step 6: Merge the heads (concatenate the last head_dim axis)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        print(f"values.size(): {values.size()}")

        # Step 7: Final linear projection to match d_model
        out = self.linear_layer(values)
        print(f"out.size(): {out.size()}")
        return out
    








# Model/inputs setup
input_dim = 1024   # Input feature size per token
d_model = 512      # Embedding/model size (must divide num_heads)
num_heads = 8
batch_size = 30
sequence_length = 5

# Create random input
x1 = torch.randn((batch_size, sequence_length, input_dim))
x2 = torch.randn((batch_size, sequence_length, input_dim))

# Instantiate MultiheadAttention class and run
model = MHA_1(d_model, num_heads)
output1 = model.forward(x1)
print(output1)

model = MHA_2(input_dim, d_model, num_heads)
output2 = model.forward(x2)
print(output2)