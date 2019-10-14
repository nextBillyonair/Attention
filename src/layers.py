import torch
from torch.nn import Module, TransformerEncoderLayer, TransformerEncoder


class BatchFirstTransformer(Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = TransformerEncoderLayer(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_t = src.transpose(0, 1)
        encodings = self.encoder(src_t, src_mask=src_mask,
                                 src_key_padding_mask=src_key_padding_mask)
        return encodings.transpose(1, 0)

# BATCH FIRST
class StackedTransformer(Module):

    def __init__(self, embedding_size=64, n_heads=1, hidden_size=64, n_layers=6, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(embedding_size, n_heads, hidden_size, dropout)
        self.stacked_transformer = TransformerEncoder(encoder_layer, n_layers)
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.encoder_hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_t = src.transpose(0, 1)
        encodings = self.stacked_transformer(src_t, src_mask=src_mask,
                                             src_key_padding_mask=src_key_padding_mask)
        return encodings.transpose(1, 0)
