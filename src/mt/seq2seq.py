import torch
from torch import nn
from typing_extensions import override

EncoderState = tuple[torch.Tensor, torch.Tensor]
DecoderState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Encoder(nn.Module):
    @override
    def forward(
        self, x: torch.Tensor, valid_lengths: torch.Tensor | None = None
    ) -> EncoderState:
        raise NotImplementedError


class Decoder(nn.Module):
    def init_state(
        self, _encoded: EncoderState, _valid_lengths: torch.Tensor
    ) -> DecoderState:
        raise NotImplementedError

    @override
    def forward(
        self, x: torch.Tensor, state: DecoderState
    ) -> tuple[torch.Tensor, DecoderState]:
        raise NotImplementedError


class AttentionDecoder(Decoder):
    def __init__(self) -> None:
        super().__init__()

    @property
    def attention_weights(self) -> list[torch.Tensor]:
        raise NotImplementedError
