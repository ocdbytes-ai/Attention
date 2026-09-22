import torch
from torch import nn


def masked_softmax(x: torch.Tensor, valid_lengths: torch.Tensor | None) -> torch.Tensor:
    # x : shape (batch_size, num_queries, num_keys/values)
    # num_keys : each query can attend to
    # valid_lengths : A tensor description which tells valid lengths for per batch or per query
    # - if valid_lengths.dim == 1 : describes per batch length
    # - if valid_lengths.dim == 2 : describes per batch, per query lengths

    if valid_lengths is None:
        return nn.functional.softmax(x, dim=1)
    else:
        shape = x.shape
        if valid_lengths.dim() == 1:
            valid_lengths = torch.repeat_interleave(valid_lengths, shape[1])
        else:
            valid_lengths = valid_lengths.reshape(-1)

        x = sequence_mask(x.reshape(-1, shape[-1]), valid_lengths, -1e6)
        return nn.functional.softmax(x.reshape(shape), dim=-1)


def sequence_mask(
    x: torch.Tensor, valid_lengths: torch.Tensor, value: float = 0.0
) -> torch.Tensor:
    # get the number of queries
    max_length = x.size(1)
    # Add a dimension in front
    positions = torch.arange(max_length, device=x.device).unsqueeze(0)
    lengths = valid_lengths.unsqueeze(1)
    mask = positions < lengths
    x[~mask] = value
    return x
