import torch
from torch.nn import Module, TransformerEncoderLayer


class BatchFirstTransformer(Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = TransformerEncoderLayer(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_t = src.transpose(0, 1)
        encodings = self.encoder(src_t, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return encodings.transpose(1, 0)
