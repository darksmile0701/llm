import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # Создаём dim//2 частот
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (end, dim//2)
    
    # Дублируем частоты, чтобы получить размерность (end, dim)
    # Это нужно, чтобы cos/sin совпадали по размеру с q и k
    freqs = torch.cat([freqs, freqs], dim=-1)  # (end, dim)
    
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # Добавляем размерности для batch и heads: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False): 
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        # Кэш для cos и sin
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        
        # Маска для causal attention
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def _update_cos_sin_cache(self, seq_len):
        """Обновляет кэш позиционных частот при необходимости"""
        if self.cos_cached is None or seq_len > self.cos_cached.shape[0]:
            # ВАЖНО: передаём self.head_dim, а не d_in!
            cos, sin = precompute_freqs_cis(self.head_dim, seq_len * 2)
            self.cos_cached = cos.to(device=self.W_query.weight.device)
            self.sin_cached = sin.to(device=self.W_query.weight.device)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # Убедимся, что кэш достаточного размера
        self._update_cos_sin_cache(num_tokens)
        
        # Получаем cos и sin для текущей длины последовательности
        cos = self.cos_cached[:num_tokens]
        sin = self.sin_cached[:num_tokens]

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Разделяем на головы: (batch, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # Транспонируем для умножения: (batch, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # ПРИМЕНЯЕМ RoPE К QUERY И KEY
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec