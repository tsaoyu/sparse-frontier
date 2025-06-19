import torch
import torch.nn.functional as F
from .abstract_attention import AbstractAttention


class RLSparseAttention(AbstractAttention):
    """Prototype reinforcement learning based sparse attention."""

    def __init__(self, epsilon: float = 0.1, keep_ratio: float = 0.5, lr: float = 0.01):
        super().__init__()
        self.epsilon = epsilon
        self.keep_ratio = keep_ratio
        self.lr = lr
        self.q_tables: dict[int, torch.Tensor] = {}
        self.training = True

    def _ensure_q_table(self, layer_idx: int, seq_len: int, num_kv_heads: int, device: torch.device):
        if layer_idx not in self.q_tables:
            self.q_tables[layer_idx] = torch.zeros(num_kv_heads, seq_len, device=device)
        elif self.q_tables[layer_idx].size(1) < seq_len:
            extra = seq_len - self.q_tables[layer_idx].size(1)
            pad = torch.zeros(num_kv_heads, extra, device=device)
            self.q_tables[layer_idx] = torch.cat([self.q_tables[layer_idx], pad], dim=1)

    def _select_indices(self, q_table: torch.Tensor, n_keep: int) -> list[torch.Tensor]:
        indices = []
        for head_scores in q_table:
            if torch.rand(1).item() < self.epsilon:
                sel = torch.randperm(head_scores.numel(), device=head_scores.device)[:n_keep]
            else:
                sel = torch.topk(head_scores, n_keep).indices
            sel, _ = torch.sort(sel)
            indices.append(sel)
        return indices

    def __call__(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        batch, num_q, seq_len, head_dim = queries.shape
        _, num_kv, _, _ = keys.shape
        self._ensure_q_table(layer_idx, seq_len, num_kv, queries.device)
        q_table = self.q_tables[layer_idx][:, :seq_len]
        n_keep = max(1, int(seq_len * self.keep_ratio))
        selected_indices = self._select_indices(q_table, n_keep)
        group = num_q // num_kv
        output = torch.zeros_like(queries)

        rewards = []
        for kv_idx in range(num_kv):
            sel = selected_indices[kv_idx]
            k_sel = keys[:, kv_idx, sel, :]
            v_sel = values[:, kv_idx, sel, :]
            causal = sel.unsqueeze(0) > torch.arange(seq_len, device=sel.device).unsqueeze(1)
            for g in range(group):
                q_head = kv_idx * group + g
                q = queries[:, q_head, :, :]
                out = F.scaled_dot_product_attention(q, k_sel, v_sel, attn_mask=causal)
                output[:, q_head, :, :] = out
                rewards.append(out.abs().mean())
            if self.training:
                reward = torch.stack(rewards).mean().detach()
                q_table[kv_idx, sel] += self.lr * (reward - q_table[kv_idx, sel])
        sparsity = 1.0 - n_keep / seq_len
        self.layer_sparsity_statistics.append(torch.tensor(sparsity, device=queries.device))
        return output

    def decode(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        output: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        num_kv, num_blocks, block_size, head_dim = k_cache.shape
        seq_len = cache_seqlens[0].item()
        k_flat = k_cache.view(num_kv, -1, head_dim)[:, :seq_len]
        v_flat = v_cache.view(num_kv, -1, head_dim)[:, :seq_len]
        self._ensure_q_table(layer_idx, seq_len, num_kv, query.device)
        q_table = self.q_tables[layer_idx][:, :seq_len]
        n_keep = max(1, int(seq_len * self.keep_ratio))
        selected_indices = self._select_indices(q_table, n_keep)
        group = query.shape[1] // num_kv
        rewards = []
        for kv_idx in range(num_kv):
            sel = selected_indices[kv_idx]
            k_sel = k_flat[kv_idx, sel]
            v_sel = v_flat[kv_idx, sel]
            causal = sel.unsqueeze(0) > (seq_len - 1)
            for g in range(group):
                q_head = kv_idx * group + g
                q = query[:, q_head, :].unsqueeze(1)
                out = F.scaled_dot_product_attention(q, k_sel.unsqueeze(0), v_sel.unsqueeze(0), attn_mask=causal)
                output[:, q_head, :] = out.squeeze(1)
                rewards.append(out.abs().mean())
            if self.training:
                reward = torch.stack(rewards).mean().detach()
                q_table[kv_idx, sel] += self.lr * (reward - q_table[kv_idx, sel])

    def kv_compress(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        seq_len = keys.size(0)
        num_kv = keys.size(1)
        self._ensure_q_table(0, seq_len, num_kv, queries.device)
        q_table = self.q_tables[0][:, :seq_len]
        n_keep = max(1, int(seq_len * self.keep_ratio))
        selected_indices = self._select_indices(q_table, n_keep)
        compressed_keys = []
        compressed_values = []
        for kv_idx in range(num_kv):
            sel = selected_indices[kv_idx]
            compressed_keys.append(keys[sel, kv_idx])
            compressed_values.append(values[sel, kv_idx])
        compressed_keys = torch.stack(compressed_keys)
        compressed_values = torch.stack(compressed_values)
        seq_lens = torch.full((num_kv,), n_keep, device=queries.device, dtype=torch.long)
        sparsity = 1.0 - n_keep / seq_len
        self.layer_sparsity_statistics.append(torch.tensor(sparsity, device=queries.device))
        return compressed_keys, compressed_values, seq_lens
