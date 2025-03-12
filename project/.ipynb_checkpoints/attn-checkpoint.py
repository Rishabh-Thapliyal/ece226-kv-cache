import torch
from model_args import ModelArgs
from emb import apply_rotary_emb

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CustomAttention(nn.Module):
    '''MultiHead Attention layer'''

    def __init__(self, args: ModelArgs):
        '''
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of Multi-heads
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.
        '''
        super().__init__()  # Initialise the parent class

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads // 1
        self.n_local_kv_heads = self.n_kv_heads // 1
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // self.n_kv_heads
        self.is_training = args.is_training
        self.is_kv_cache = args.is_kv_cache
        # self.device = args.device

        self.wq = nn.Linear(
            in_features=args.dim,
            out_features=args.dim,
            bias=False,
            device=args.device,
        )
        self.wk = nn.Linear(
            in_features=args.dim,
            out_features=args.dim,
            bias=False,
            device=args.device,
        )
        self.wv = nn.Linear(
            in_features=args.dim,
            out_features=args.dim,
            bias=False,
            device=args.device,
        )
        self.wo = nn.Linear(
            in_features=args.dim,
            out_features=args.dim,
            bias=False,
            device=args.device,
        )

        # KV cache for inference
        self.cache_k = torch.zeros(
            size=(
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            ),
            device=args.device,
            requires_grad=False,
        )
        self.cache_v = torch.zeros(
            size=(
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            ),
            device=args.device,
            requires_grad=False,
        )

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        bsz, seqlen, _ = x.shape  # _ --> args.dim

        # Linear transformations for queries, keys, and values
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape queries, keys, and values for multi-head attention
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply rotary embeddings (if needed)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if not self.is_training and self.is_kv_cache:  # Inference mode (use KV cache)
            # Update KV cache
            self.cache_k = self.cache_k.to(xq)  # Ensure cache is on the same device as xq
            self.cache_v = self.cache_v.to(xq)
            self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

            # Retrieve cached keys and values
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:  # Training mode (no KV cache)
            keys = xk
            values = xv

        # Repeat keys and values if n_kv_heads < n_heads
        # keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # Transpose for multi-head attention
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # Compute attention scores
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim)  # (bs, n_local_heads, seqlen, seqlen)
        if mask is not None:
            scores = scores + mask  # Apply mask (causal or otherwise)

        # Softmax and attention output
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # (bs, n_local_heads, seqlen, seqlen)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)

        # if output.size() == xq.size():
        #     print("here")

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # (bs, seqlen, dim)
        return self.wo(output)