import math
from dataclasses import dataclass

import torch
from torch import nn

from .ops import masked_softmax


@dataclass
class DotProductAttentionInputs:
    dropout: float


class DotProductAttention(nn.Module):
    def __init__(self, dot_product_attention_inputs: DotProductAttentionInputs) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dot_product_attention_inputs.dropout)
        self.attention_weights = torch.empty(0)

    def forward(
        self,
        # (batch_size, num_queries, len_query/key)
        queries: torch.Tensor,
        # (batch_size, num_key_val_pairs, len_query/key)
        keys: torch.Tensor,
        # (batch_size, num_key_val_pairs, len_value)
        values: torch.Tensor,
        # (batch_size) || (batch_size, num_queries)
        valid_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        len_query = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        # For shape matching
        # (batch_size, num_key_val_pairs, len_query/key) -> (batch_size, len_query/keys, num_key_val_pairs)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(len_query)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


@dataclass
class AdditiveAttentionInputs:
    num_hiddens: int
    dropout: float


class AdditiveAttention(nn.Module):
    def __init__(self, additive_attention_inputs: AdditiveAttentionInputs):
        super().__init__()
        self.W_k = nn.LazyLinear(additive_attention_inputs.num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(additive_attention_inputs.num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(additive_attention_inputs.dropout)
        self.attention_weights = torch.empty(0)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lens: torch.Tensor,
    ) -> torch.Tensor:
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion,
        # shape of queries: (batch_size, num_queries, 1, num_hiddens)
        # shape of keys: (batch_size, 1, num_key_val_pairs, num_hiddens)
        # Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape.
        # Shape of scores: (batch_size, num_queries, num_key_val_pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, num_key_val_pairs, len_values)
        return torch.bmm(self.dropout(self.attention_weights), values)
