class DistributedAttention(nn.Module):
    def __init__(self, Wq, Wk, Wv, Wo, num_heads, workers):
        self.Wq, self.Wk, self.Wv, self.Wo = Wq, Wk, Wv, Wo
        self.num_heads = num_heads
        self.workers = workers  # list of worker endpoints

    def forward(self, x):
        # 1. Local projections
        Q = x @ self.Wq.T   # [B, S, D]
        K = x @ self.Wk.T
        V = x @ self.Wv.T

        # 2. Reshape to heads: [B, S, H, D/H]
        Qh, Kh, Vh = split_into_heads(Q, K, V, self.num_heads)

        # 3. Send each head slice to a worker
        futures = []
        for i, worker in enumerate(self.workers):
            payload = {"Qh": Qh[i], "Kh": Kh[i], "Vh": Vh[i]}
            futures.append(send_async(worker, payload))

        # 4. Wait for results
        out_heads = [recv(f) for f in futures]

        # 5. Concatenate results along head dimension
        out = concat_heads(out_heads)  # [B, S, D]

        # 6. Output projection
        return out @ self.Wo.T




def worker_loop():
    while True:
        payload = recv_from_manager()
        Qh, Kh, Vh = payload["Qh"], payload["Kh"], payload["Vh"]

        # Local scaled dot-product attention
        scores = softmax((Qh @ Kh.T) / sqrt(Qh.shape[-1]))  # [S, S]
        out_h = scores @ Vh  # [S, head_dim]

        send_to_manager(out_h)






class BandwidthCostCalculator:
    def __init__(self, dtype_size_bytes=4):  # float32 = 4 bytes
        self.dtype_size = dtype_size_bytes

    def attention_split_cost(self, batch, seq_len, d_model, num_heads, num_workers):
        head_dim = d_model // num_heads
        # Per worker payload: Qh, Kh, Vh
        per_worker_send = batch * seq_len * head_dim * 3
        per_worker_recv = batch * seq_len * head_dim
        total_send = per_worker_send * num_workers
        total_recv = per_worker_recv * num_workers
        return (total_send + total_recv) * self.dtype_size

    def mlp_split_cost(self, batch, seq_len, d_in, d_hidden, num_workers):
        # Assuming column-parallel W1 and row-parallel W2
        # Send X once to all workers, receive partial sums
        send = batch * seq_len * d_in * num_workers
        recv = batch * seq_len * d_in * num_workers
        return (send + recv) * self.dtype_size





other:


from sam2.modeling.sam2_base import SAM2Base
model = SAM2Base(...)




for name, module in model.named_modules():
    print(name, type(module))




from my_distributed.attn import DistributedAttention

def replace_attention(model, workers):
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            setattr(model, name, DistributedAttention(
                Wq=module.in_proj_weight[:d], 
                Wk=module.in_proj_weight[d:2*d], 
                Wv=module.in_proj_weight[2*d:], 
                Wo=module.out_proj.weight, 
                num_heads=module.num_heads,
                workers=workers
            ))
        else:
            replace_attention(module, workers)

replace_attention(model, workers=["worker1:5000", "worker2:5000"])
