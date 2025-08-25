"""Collection of the different functional forms of attention."""

import torch as T
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func
except ImportError:
    # Flash Attention을 사용할 수 없는 경우 대체 구현
    def flash_attn_varlen_kvpacked_func(q, kv, culens_q, culens_k, maxlen_q, maxlen_k, dropout_p=0.0, causal=False):
        """기본 PyTorch Cross Attention으로 대체 구현"""
        import torch
        import torch.nn.functional as F
        
        # q는 (total_len_q, num_heads, head_dim)
        # kv는 (total_len_k, 2, num_heads, head_dim) 형태
        total_len_q, num_heads, head_dim = q.shape
        total_len_k, _, _, _ = kv.shape
        
        # K, V 분리
        k = kv[:, 0]  # (total_len_k, num_heads, head_dim)
        v = kv[:, 1]  # (total_len_k, num_heads, head_dim)
        
        # 배치별로 분리
        outputs = []
        start_q = 0
        start_k = 0
        
        for len_q, len_k in zip(culens_q, culens_k):
            end_q = start_q + len_q
            end_k = start_k + len_k
            
            # 현재 배치의 Q, K, V 추출
            q_batch = q[start_q:end_q].transpose(0, 1)  # (num_heads, seq_len_q, head_dim)
            k_batch = k[start_k:end_k].transpose(0, 1)  # (num_heads, seq_len_k, head_dim)
            v_batch = v[start_k:end_k].transpose(0, 1)  # (num_heads, seq_len_k, head_dim)
            
            # Scaled dot-product attention
            scale = (head_dim ** -0.5)
            attn = torch.bmm(q_batch, k_batch.transpose(1, 2)) * scale
            
            if causal:
                # Cross attention에서는 일반적으로 causal mask를 사용하지 않지만
                # 필요한 경우 구현
                pass
            
            attn = F.softmax(attn, dim=-1)
            
            if dropout_p > 0.0:
                attn = F.dropout(attn, p=dropout_p, training=True)
            
            out = torch.bmm(attn, v_batch)  # (num_heads, seq_len_q, head_dim)
            out = out.transpose(0, 1)  # (seq_len_q, num_heads, head_dim)
            
            outputs.append(out)
            start_q = end_q
            start_k = end_k
        
        # 결과 연결
        result = torch.cat(outputs, dim=0)
        return result
    
    def flash_attn_varlen_qkvpacked_func(qkv, culens, maxlen, dropout_p=0.0, causal=False):
        """기본 PyTorch Attention으로 대체 구현"""
        import torch
        import torch.nn.functional as F
        
        # qkv는 (total_len, 3, num_heads, head_dim) 형태
        total_len, _, num_heads, head_dim = qkv.shape
        
        # Q, K, V 분리
        q = qkv[:, 0]  # (total_len, num_heads, head_dim)
        k = qkv[:, 1]
        v = qkv[:, 2]
        
        # 배치별로 분리
        outputs = []
        start = 0
        for length in culens:
            end = start + length
            
            # 현재 배치의 Q, K, V 추출
            q_batch = q[start:end].transpose(0, 1)  # (num_heads, seq_len, head_dim)
            k_batch = k[start:end].transpose(0, 1)
            v_batch = v[start:end].transpose(0, 1)
            
            # Scaled dot-product attention
            scale = (head_dim ** -0.5)
            attn = torch.bmm(q_batch, k_batch.transpose(1, 2)) * scale
            
            if causal:
                mask = torch.triu(torch.ones(length, length), diagonal=1).bool()
                attn.masked_fill_(mask.unsqueeze(0), float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            
            if dropout_p > 0.0:
                attn = F.dropout(attn, p=dropout_p, training=True)
            
            out = torch.bmm(attn, v_batch)  # (num_heads, seq_len, head_dim)
            out = out.transpose(0, 1)  # (seq_len, num_heads, head_dim)
            
            outputs.append(out)
            start = end
        
        # 결과 연결
        result = torch.cat(outputs, dim=0)
        return result


def flash_self_attention(
    x: T.Tensor,
    culens: T.Tensor,
    maxlen: int,
    drop: float,
    causal: bool,
    weight: T.Tensor,
    bias: T.Tensor,
    num_heads: int,
) -> T.Tensor:
    dim = x.size(-1)
    qkv = F.linear(x, weight, bias)
    qkv = qkv.view(-1, 3, num_heads, dim // num_heads)
    attn = flash_attn_varlen_qkvpacked_func(qkv, culens, maxlen, drop, causal=causal)
    return attn.contiguous().view(-1, dim)


def flash_cross_attention(
    x: T.Tensor,
    culens: T.Tensor,
    maxlen: int,
    kv: T.Tensor,
    kv_culens: T.Tensor,
    kv_maxlen: int,
    drop: float,
    causal: bool,
    weight: T.Tensor,
    bias: T.Tensor | None,
    num_heads: int,
) -> T.Tensor:
    dim = x.size(-1)
    head_dim = dim // num_heads
    w_q, w_kv = weight.split([dim, dim * 2])
    b_q, b_kv = bias.split([dim, dim * 2]) if bias is not None else (None, None)
    q_proj = F.linear(x, w_q, b_q).view(-1, num_heads, head_dim)
    kv_proj = F.linear(kv, w_kv, b_kv).view(-1, 2, num_heads, head_dim)
    attn = flash_attn_varlen_kvpacked_func(
        q_proj, kv_proj, culens, kv_culens, maxlen, kv_maxlen, drop, causal=causal
    )
    return attn.contiguous().view(-1, dim)
