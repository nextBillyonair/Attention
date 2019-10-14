import torch
from torch.nn import Module
import math

class PositionalEmbeddings(Module):

    def __init__(self, feature_len, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.seq_len = max_seq_len
        self.feature_len = feature_len
        self.dropout = Dropout(p=dropout)

        # precompute
        pos = torch.arange(start=0, end=self.seq_len).unsqueeze(1).float()
        i = torch.arange(0, self.feature_len, 2).float()
        h = (pos.log() - i / self.feature_len * math.log(10000)).exp()
        self.pe = torch.empty(self.seq_len, self.feature_len)
        self.pe[:, 0::2] = torch.sin(h)
        self.pe[:, 1::2] = torch.cos(h)

    def forward(self, encodings):
        seq_len = encodings.size(1)
        encodings = encodings + self.pe[:seq_len, :]
        return self.dropout(encodings)


class SineEmbeddings(Module):

    def __init__(self, feature_len, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.seq_len = max_seq_len
        self.feature_len = feature_len
        self.dropout = Dropout(p=dropout)

        # precompute
        pos = torch.arange(start=0, end=self.seq_len).unsqueeze(1).float()
        i = torch.arange(0, self.feature_len, 1).float()
        h = (pos.log() - i / self.feature_len * math.log(10000)).exp()
        self.pe = torch.sin(h)

    def forward(self, encodings):
        seq_len = encodings.size(1)
        encodings = encodings + self.pe[:seq_len, :]
        return self.dropout(encodings)


class CosineEmbeddings(Module):

    def __init__(self, feature_len, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.seq_len = max_seq_len
        self.feature_len = feature_len
        self.dropout = Dropout(p=dropout)

        # precompute
        pos = torch.arange(start=0, end=self.seq_len).unsqueeze(1).float()
        i = torch.arange(0, self.feature_len, 1).float()
        h = (pos.log() - i / self.feature_len * math.log(10000)).exp()
        self.pe = torch.cos(h)

    def forward(self, encodings):
        seq_len = encodings.size(1)
        encodings = encodings + self.pe[:seq_len, :]
        return self.dropout(encodings)
