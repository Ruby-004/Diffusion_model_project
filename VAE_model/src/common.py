import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        in_proj_bias = True,
        out_proj_bias = True
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.in_proj_bias = in_proj_bias
        self.out_proj_bias = out_proj_bias

        self.head_dim = self.embed_dim // self.num_heads

        self.in_proj = nn.Linear(
            embed_dim, 3*embed_dim, bias=self.in_proj_bias
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=self.out_proj_bias
        )

    def forward(self, x: torch.Tensor, mask=False):
        
        in_shape = x.shape
        b, seq_len, d = in_shape

        # query, key, value. each of shape (batch, seq_len, embed_dim)
        query, key, value = self.in_proj(x).chunk(3, dim=-1)

        tmp_shape = (b, seq_len, self.num_heads, self.head_dim)

        # 1. (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim)
        # 2. (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        query = query.view(tmp_shape).transpose(1, 2)
        key = key.view(tmp_shape).transpose(1, 2)
        value = value.view(tmp_shape).transpose(1, 2)

        # (batch, num_heads, seq_len, seq_len)
        weight = query @ key.transpose(-1, -2)

        if mask:
            # upper triangular matrix filled with 1
            mask_tensor = torch.ones_like(
                weight, dtype=torch.bool, device=weight.device
            ).triu(diagonal=1)

            weight.masked_fill_(mask_tensor, -torch.inf)


        weight /= math.sqrt(self.head_dim)
        weight = F.softmax(weight, dim=-1)

        # (batch, num_heads, seq_len, head_dim)
        out = weight @ value

        # (batch, seq_len, num_heads, head_dim)
        out = out.transpose(1, 2)

        # (batch, seq_len, embed_dim)
        out = out.reshape(in_shape)

        out = self.out_proj(out)
        return out



def get_padding(kernel_size: int):
    """Get padding."""

    if (kernel_size % 2) == 0:
        padding = (kernel_size // 2) - 1
    else:
        padding = kernel_size // 2
    
    return padding