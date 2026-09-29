import torch

from src.mt.model import (
    Seq2SeqAttentionDecoder,
    Seq2SeqAttentionDecoderInputs,
    Seq2SeqEncoder,
    Seq2SeqInputs,
)


def test_bidirectional_encoder_decoder_shapes() -> None:
    encoder = Seq2SeqEncoder(Seq2SeqInputs(20, 6, 8, 2))
    decoder = Seq2SeqAttentionDecoder(
        Seq2SeqAttentionDecoderInputs(30, 6, 8, 2)
    )
    source = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    source_lengths = torch.tensor([3, 2])

    encoded = encoder(source, source_lengths)
    assert encoder.gru.bidirectional
    assert encoded[0].shape == (2, 4, 8)
    assert encoded[1].shape == (2, 2, 8)

    logits, _ = decoder(
        torch.tensor([[1, 2, 3], [1, 4, 5]]),
        decoder.init_state(encoded, source_lengths),
    )
    assert logits.shape == (2, 3, 30)


if __name__ == "__main__":
    test_bidirectional_encoder_decoder_shapes()
