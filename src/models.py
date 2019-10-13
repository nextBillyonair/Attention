import torch
from torch.nn import Module, Linear, Embedding
from attention import (
    ConcatAttention, DotAttention,
    MeanAttention, LastInSeqAttention
)
from layers import BatchFirstTransformer


# Define the model architecture with embedding layers and LSTM layers:
class RNNModel(nn.Module):
    def __init__(self, embedding_matrix, recurrent_type='LSTM', num_layers=1, rnn_hidden_size=64,
                 attn_hidden_size=64, linear_hidden_size=32, attention_type=ConcatAttention, num_outputs=5):
        super().__init__()
        self.recurrent_type = recurrent_type
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.rnn = getattr(nn, recurrent_type)(embedding_matrix.size(-1), rnn_hidden_size,
                    num_layers=num_layers, bidirectional=False, batch_first=True)
        self.attention = attention_type(rnn_hidden_size, attn_hidden_size)
        self.linear = nn.Linear(attn_hidden_size, linear_hidden_size)
        self.out = nn.Linear(linear_hidden_size, num_outputs)

    def forward(self, x, return_weights=True):
        h_embedding = self.embedding(x)
        h_rnn, final_hidden_states = self.rnn(h_embedding)

        if self.recurrent_type == 'LSTM':
            context = final_hidden_states[1]
        else:
            context = final_hidden_states

        context = context[-1].unsqueeze(-2).expand_as(h_rnn)
        context_vec, weights = self.attention(h_rnn, context=context, mask=(x == 0.))
        h_linear = self.dropout(context_vec)
        out = self.out(h_linear)

        if return_weights:
            return out, weights
        return out


# Define the model architecture with embedding layers and LSTM layers:
class TransformerModel(nn.Module):
    def __init__(self, embedding_matrix, num_layers=1, transformer_hidden_size=64, nheads=5,
                 attn_hidden_size=64, attention_type=ConcatAttention,
                 embed_size=64, dropout=0.1, num_outputs=5,
                ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.lin2 = nn.Linear(embedding_matrix.size(-1), embed_size)
        self.self_attn = BatchFirstTransformer(embed_size, nhead=nheads,
                    dim_feedforward=transformer_hidden_size, activation="relu")
        self.attention = attention_type(embed_size, attn_hidden_size)
        self.out = nn.Linear(embed_size, num_outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_weights=True):
        h_embedding = self.lin2(self.embedding(x))
        h_rnn = self.self_attn(h_embedding, src_key_padding_mask=(x == 0.))
        context_vec, weights = self.attention(h_rnn, mask=(x == 0.))
        context_vec = self.dropout(context_vec)
        out = self.out(context_vec)

        if return_weights:
            return out, weights
        return out
