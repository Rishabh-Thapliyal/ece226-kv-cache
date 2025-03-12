import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# def _initialize_affine_weight(
#     weight: torch.Tensor,
#     out_features: int,
#     in_features: int,
#     per_partition_size: int,
#     partition_dim: int,
#     init_method: Callable[[torch.Tensor], torch.Tensor],
#     stride: int = 1,
#     return_master_weight: bool = False,
# ) -> Optional[torch.Tensor]:
#     """Initialize affine weight for model parallel.
#
#     Build the master weight on all processes and scatter
#     the relevant chunk."""
#
#     # If we only use 1 process for model parallelism, bypass scatter.
#     world_size = get_model_parallel_world_size()
#     if world_size == 1:
#         init_method(weight)
#         if return_master_weight:
#             return weight
#         return None


class Embedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: Optional[int] = None,
            max_norm: Optional[float] = None,
            norm_type: float = 2.0,
            scale_grad_by_freq: bool = False,
            sparse: bool = False,
            init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
            device : str = "cpu",
            keep_master_weight_for_test: bool = False,
    ) -> None:
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        # Divide the weight matrix along the embedding dimension.
        # world_size = get_model_parallel_world_size()
        # self.embedding_dim_per_partition = divide_and_check_no_remainder(self.embedding_dim, world_size)

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
        # And initialize.

        self.weight = init_method(self.weight)
        self.device = device
        # _initialize_affine_weight(
        #     self.weight,
        #     self.num_embeddings,
        #     self.embedding_dim,
        #     self.embedding_dim_per_partition,
        #     1,
        #     init_method,
        #     stride=1,
        #     return_master_weight=False,
        # )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # input_parallel = copy_to_model_parallel_region(input_)
        # print(torch.all(self.weight == 0).item())
        output = F.embedding(
            input_,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        ).to(self.device)
        # output = gather_from_model_parallel_region(output_parallel)
        return output


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

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