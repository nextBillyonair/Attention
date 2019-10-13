import torch
from torch.nn import Module
import math

class SineEmbeddings(Module):

    def __init__(self, max_seq_len, feature_len):
        super().__init__()
        self.seq_len = max_seq_len
        self.feature_len = feature_len

        # precompute
        pos = torch.arange(start=0, end=self.seq_len).unsqueeze(1).float()#.expand_as(encodings)
        i = torch.arange(0, self.feature_len, 1).unsqueeze(0).float()
        h = (pos.log() - 2 * i / self.feature_len * math.log(10000)).exp()
        self.pe = torch.sin(h).unsqueeze(0)

    def forward(self, encodings):
        seq_len = encodings.size(1)
        return encodings + self.pe[:, :seq_len, :]


class CosineEmbeddings(Module):

    def __init__(self, max_seq_len, feature_len):
        super().__init__()
        self.seq_len = max_seq_len
        self.feature_len = feature_len

        # precompute
        pos = torch.arange(start=0, end=self.seq_len).unsqueeze(1).float()#.expand_as(encodings)
        i = torch.arange(0, self.feature_len, 1).unsqueeze(0).float()
        h = (pos.log() - 2 * i / self.feature_len * math.log(10000)).exp()
        self.pe = torch.cos(h).unsqueeze(0)

    def forward(self, encodings):
        seq_len = encodings.size(1)
        return encodings + self.pe[:, :seq_len, :]


class PositionalEmbeddings(Module):

    def __init__(self, max_seq_len, feature_len):
        super().__init__()
        self.seq_len = max_seq_len
        self.feature_len = feature_len

        # precompute
        pos = torch.arange(start=0, end=self.seq_len).unsqueeze(1).float()#.expand_as(encodings)
        i = torch.arange(0, self.feature_len, 1).unsqueeze(0).float()
        h = (pos.log() - 2 * i / self.feature_len * math.log(10000)).exp()
        self.pe = torch.empty(1, self.seq_len, self.feature_len)
        self.pe[:, :, 0::2] = torch.sin(h).unsqueeze(0)[:, :, 0::2]
        self.pe[:, :, 1::2] = torch.cos(h).unsqueeze(0)[:, :, 1::2]

    def forward(self, encodings):
        seq_len = encodings.size(1)
        return encodings + self.pe[:, :seq_len, :]
