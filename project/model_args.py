from typing import Callable, Optional, Tuple

class ModelArgs:

    # def __init__(self, vocab, device):
    dim: int = 512
    n_layers: int = 2
    n_heads: int = 2
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    is_training: bool = True
    is_kv_cache: bool = True
    device: str = None