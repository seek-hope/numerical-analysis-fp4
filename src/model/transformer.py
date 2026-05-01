"""Micro-Gemma-FP: ~164M Transformer with RMSNorm, RoPE, GQA, sliding/full attention.

Supports FP8/FP4 quantization via forward hooks during QAT and PTQ.
Architecture mirrors Gemma 4 features at a research-friendly scale.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import MicroGemmaFPConfig


# ═══════════════════════════════════════════════════════════════
# RMS Normalization
# ═══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ═══════════════════════════════════════════════════════════════
# Rotary Position Embedding
# ═══════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        seq_len = x.shape[1]
        device = x.device
        t = position_ids.float()
        freqs = torch.outer(t.flatten(), self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        return cos.to(device), sin.to(device)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    x_rot = x.float()
    x1, x2 = x_rot.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return (x_rot * cos + x_rotated * sin).to(x.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


# ═══════════════════════════════════════════════════════════════
# Attention
# ═══════════════════════════════════════════════════════════════

class Attention(nn.Module):
    def __init__(self, config: MicroGemmaFPConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim if self.layer_type == 'sliding' else config.global_head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size + config.hidden_size_per_layer_input,
                                self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size + config.hidden_size_per_layer_input,
                                self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size + config.hidden_size_per_layer_input,
                                self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, pl_emb, position_ids, attention_mask=None):
        batch, seq_len, _ = hidden_states.shape
        x = torch.cat([hidden_states, pl_emb], dim=-1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self.rotary(q, position_ids)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=True)
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(attn_output)


# ═══════════════════════════════════════════════════════════════
# Feed-Forward Network
# ═══════════════════════════════════════════════════════════════

class FFN(nn.Module):
    def __init__(self, config: MicroGemmaFPConfig):
        super().__init__()
        in_dim = config.hidden_size + config.hidden_size_per_layer_input
        self.gate_proj = nn.Linear(in_dim, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_dim, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, pl_emb):
        x = torch.cat([hidden_states, pl_emb], dim=-1)
        gate = F.gelu(self.gate_proj(x), approximate='tanh')
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ═══════════════════════════════════════════════════════════════
# Transformer Layer
# ═══════════════════════════════════════════════════════════════

class TransformerLayer(nn.Module):
    def __init__(self, config: MicroGemmaFPConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = Attention(config, layer_idx)
        self.ffn = FFN(config)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, pl_emb, position_ids, attention_mask=None):
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.attention(hidden_states, pl_emb, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states, pl_emb)
        hidden_states = residual + hidden_states

        return hidden_states


# ═══════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════

class MicroGemmaFPModel(nn.Module):
    def __init__(self, config: MicroGemmaFPConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.per_layer_embeddings = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.hidden_size_per_layer_input)
            for _ in range(config.num_hidden_layers)
        ])
        self.layers = nn.ModuleList([
            TransformerLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            pl_emb = self.per_layer_embeddings[i](input_ids)
            hidden_states = layer(hidden_states, pl_emb, position_ids, attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class MicroGemmaFPForCausalLM(nn.Module):
    def __init__(self, config: MicroGemmaFPConfig):
        super().__init__()
        self.config = config
        self.model = MicroGemmaFPModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1))
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def get_quantizable_weights(self) -> list[tuple[str, nn.Parameter]]:
        """Return list of (name, param) for weights that can be quantized."""
        result = []
        for name, param in self.named_parameters():
            if param.dim() >= 2 and any(k in name for k in ('proj', 'embed_tokens', 'lm_head')):
                result.append((name, param))
        return result
