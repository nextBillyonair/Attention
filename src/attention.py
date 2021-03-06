import torch
from torch.nn import Module, Linear, Bilinear
from torch.nn import MultiheadAttention as TorchMultiheadAttention
import math

'''
Notes:
Masks should be 1 where we ignore
0 for keep,

X == 0., where 0 is the pad
'''


class ConcatAttention(Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.W_a = Linear(2 * embedding_size, hidden_size)
        self.v_a = Linear(hidden_size, 1, bias=False)

    def forward(self, encodings, context=None, mask=None, return_weights=True):
        batch_size, seq_len, _ = encodings.size()
        # encodings.shape == (batch_size, seq_len, embedding_dim)
        # mask.shape == (batch_size, seq_len) BOOL: 1 == PAD

        if context is None:
            context = torch.zeros_like(encodings)
        else:
            context = context[-1].unsqueeze(-2).expand_as(encodings)
        # context.shape == (batch_size, seq_len, embedding_size) after expand

        scores = torch.tanh(self.W_a(torch.cat((encodings, context), dim=-1)))
        scores = self.v_a(scores)
        # scores.shape == (batch_size, seq_len, 1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1).bool(), -1e10)

        weights = torch.softmax(scores, dim=1)
        weights = weights.view(batch_size, 1, seq_len)

        attended = torch.bmm(weights, encodings).squeeze(1)
        # attended.shape == (batch_size, embedding_dim)

        if return_weights:
            return attended, weights.view(batch_size, seq_len, 1)
        return attended


class GeneralAttention(Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.W_a = Bilinear(embedding_size, embedding_size, 1)

    def forward(self, encodings, context=None, mask=None, return_weights=True):
        batch_size, seq_len, _ = encodings.size()
        # encodings.shape == (batch_size, seq_len, embedding_dim)
        # mask.shape == (batch_size, seq_len) BOOL: 1 == PAD

        if context is None:
            context = torch.ones_like(encodings)
        else:
            context = context[-1].unsqueeze(-2).expand_as(encodings)
        # context.shape == (batch_size, seq_len, embedding_size) after expand

        scores = self.W_a(encodings.contiguous(), context.contiguous())
        # scores.shape == (batch_size, seq_len, 1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1).bool(), -1e10)

        weights = torch.softmax(scores, dim=1)
        weights = weights.view(batch_size, 1, seq_len)

        attended = torch.bmm(weights, encodings).squeeze(1)
        # attended.shape == (batch_size, embedding_dim)

        if return_weights:
            return attended, weights.view(batch_size, seq_len, 1)
        return attended

class DotAttention(Module):

    def __init__(self, embedding_size, hidden_size, use_scale=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = 1 / math.sqrt(embedding_size) if use_scale else 1.

    def forward(self, encodings, context=None, mask=None, return_weights=True):
        batch_size, seq_len, _ = encodings.size()
        # encodings.shape == (batch_size, seq_len, embedding_dim)
        # mask.shape == (batch_size, seq_len) BOOL: 1 == PAD

        if context is None:
            context = torch.ones_like(encodings)
        else:
            context = context[-1].unsqueeze(-2).expand_as(encodings)
        # context.shape == (batch_size, seq_len, embedding_size) after expand

        # Reshapes to make matmul work with 4D
        encodings = encodings.view(batch_size, seq_len, 1, -1)
        context = encodings.view(batch_size, seq_len, -1, 1)
        scores = self.scale * torch.matmul(encodings, context).squeeze(-1)
        # scores.shape == (batch_size, seq_len, 1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1).bool(), -1e10)

        weights = torch.softmax(scores, dim=1)
        weights = weights.view(batch_size, 1, seq_len)
        encodings = encodings.view(batch_size, seq_len, -1) # change back

        attended = torch.bmm(weights, encodings).squeeze(1)
        # attended.shape == (batch_size, embedding_dim)

        if return_weights:
            return attended, weights.view(batch_size, seq_len, 1)
        return attended


class MeanAttention(Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()

    def forward(self, encodings, context=None, mask=None, return_weights=True):
        batch_size, seq_len, _ = encodings.size()
        # encodings.shape == (batch_size, seq_len, embedding_dim)
        # mask.shape == (batch_size, seq_len) BOOL: 1 == PAD

        if mask is not None:
            mask = 1 - mask.unsqueeze(-1).float() # reverse mask
            weights = (1 / mask.sum(dim=-2, keepdim=True)) * mask
            encodings = encodings * mask
        else:
            weights = torch.full((batch_size, seq_len, 1), 1 / seq_len)
        # weights.shape == (batch_size, seq_len, 1) # For broadcast

        attended = (encodings * weights).sum(dim=-2)
        # attended.shape == (batch_size, embedding_dim)

        if return_weights:
            return attended, weights
        return attended


class LastInSeqAttention(Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()

    def forward(self, encodings, context=None, mask=None, return_weights=True):
        # FIX MASK
        # encodings.shape == (batch_size, seq_len, embedding_dim)
        # RETURN last -> (batch_size, embedding_dim) (takes last slice in seq dim)
        if return_weights:
            return encodings[:, -1], None
        return encodings[:, -1]


# BATCH FIRST
class MultiheadAttention(Module):

    def __init__(self, embedding_size, hidden_size, nhead=1, dropout=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.nhead = nhead
        self.dropout = dropout
        self.self_attn = TorchMultiheadAttention(embedding_size, nhead, dropout=dropout)

    def forward(self, encodings, context=None, mask=None, return_weights=True):
        encodings_t = encodings.transpose(0, 1)
        attended = self.self_attn(encodings_t, encodings_t, encodings_t,
                                    attn_mask=None, key_padding_mask=mask, need_weights=False)
        attended = attended.transpose(1, 0) # BATCH, SEQ_LEN
        if return_weights:
            return attended, None
        return attended
