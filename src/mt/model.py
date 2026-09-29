from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from typing_extensions import override

from src.attention import AdditiveAttention, AdditiveAttentionInputs
from src.mt.seq2seq import AttentionDecoder, DecoderState, Encoder, EncoderState


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear) and not isinstance(module, nn.LazyLinear):
        _ = nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.GRU):
        for name, parameter in module.named_parameters():
            if "weight" in name:
                _ = nn.init.xavier_uniform_(parameter)


@dataclass
class Seq2SeqInputs:
    vocab_size: int
    embedding_size: int
    hidden_size: int
    num_layers: int
    dropout: float = 0.0


class Seq2SeqEncoder(Encoder):
    def __init__(self, inputs: Seq2SeqInputs):
        super().__init__()
        if inputs.hidden_size % 2:
            raise ValueError("hidden_size must be even for a bidirectional encoder")
        self.inputs: Seq2SeqInputs = inputs
        self.embedding: nn.Embedding = nn.Embedding(
            inputs.vocab_size, inputs.embedding_size
        )
        self.gru: nn.GRU = nn.GRU(
            inputs.embedding_size,
            inputs.hidden_size // 2,
            num_layers=inputs.num_layers,
            dropout=inputs.dropout,
            batch_first=True,
            bidirectional=True,
        )
        _ = self.apply(init_weights)

    @override
    def forward(
        self, x: torch.Tensor, valid_lengths: torch.Tensor | None = None
    ) -> EncoderState:
        embeddings = cast(torch.Tensor, self.embedding(x.long()))
        if valid_lengths is None:
            outputs, state = cast(
                tuple[torch.Tensor, torch.Tensor], self.gru(embeddings)
            )
        else:
            # Ignore padding while encoding each direction.
            packed = pack_padded_sequence(
                embeddings,
                valid_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, state = cast(
                tuple[PackedSequence, torch.Tensor], self.gru(packed)
            )
            outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=x.shape[1],
            )

        # Join each layer's forward and backward states for the decoder.
        state = state.reshape(
            self.inputs.num_layers, 2, state.shape[1], state.shape[2]
        )
        return outputs, torch.cat((state[:, 0], state[:, 1]), dim=-1)


@dataclass
class Seq2SeqAttentionDecoderInputs:
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0.0


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, inputs: Seq2SeqAttentionDecoderInputs) -> None:
        super().__init__()
        self.attention = AdditiveAttention(
            AdditiveAttentionInputs(inputs.num_hiddens, inputs.dropout)
        )
        self.embedding = nn.Embedding(inputs.vocab_size, inputs.embed_size)
        self.rnn = nn.GRU(
            inputs.embed_size + inputs.num_hiddens,
            inputs.num_hiddens,
            inputs.num_layers,
            dropout=inputs.dropout,
        )
        self.dense = nn.LazyLinear(inputs.vocab_size)
        self._attention_weights: list[torch.Tensor] = []
        _ = self.apply(init_weights)

    @override
    def init_state(
        self, encoder_outputs: EncoderState, encoder_valid_lens: torch.Tensor
    ) -> DecoderState:
        # Shape of outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = encoder_outputs
        return (outputs, hidden_state, encoder_valid_lens)

    @override
    def forward(
        self, x: torch.Tensor, state: DecoderState
    ) -> tuple[torch.Tensor, DecoderState]:
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output x: (num_steps, batch_size, embed_size)
        embeddings = self.embedding(x).permute(1, 0, 2)
        outputs: list[torch.Tensor] = []
        self._attention_weights = []
        for embedding in embeddings:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            embedding = torch.cat((context, torch.unsqueeze(embedding, dim=1)), dim=-1)
            # Reshape embedding as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(embedding.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        logits = self.dense(torch.cat(outputs, dim=0))
        return logits.permute(1, 0, 2), (
            enc_outputs,
            hidden_state,
            enc_valid_lens,
        )

    @property
    @override
    def attention_weights(self) -> list[torch.Tensor]:
        return self._attention_weights
