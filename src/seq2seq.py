import torch
import random
from torch.nn import (
    Module, GRU, Embedding, Linear
)
from .attention import ConcatAttention


class Encoder(Module):

    def __init__(self, enc_units, embedding_dim, vocab_size=10000, num_layers=1):
        super().__init__()
        self.enc_units = enc_units
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(embedding_dim, enc_units, num_layers=num_layers, batch_first=True)

    def forward(self, x, hidden=None):
        # Input.shape == (batch_size, seq_len) LONG
        x = self.embedding(x)
        # x.shape == (batch_size, seq_len, embedding_dim)
        output, state = self.gru(x, hidden)
        # output.shape == (batch_size, seq_len, enc_units)
        # state.shape == (num_layers, batch_size, enc_units)
        return output, state

    def from_pretrained(self, embedding_matrix):
        if not isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = torch.tensor(embedding_matrix).float()
        self.embedding = Embedding.from_pretrained(embedding_matrix)


class Decoder(Module):

    def __init__(self, dec_units, embedding_dim, vocab_size=10000, num_layers=1):
        super().__init__()
        self.dec_units = dec_units
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(embedding_dim, dec_units, num_layers=num_layers, batch_first=True)
        self.fc = Linear(dec_units, vocab_size)

    def forward(self, x, hidden=None, encoder_outputs=None, mask=None):
        # input.shape == (batch_size, 1)
        x = self.embedding(x)
        # x.shape == (batch_size, 1, embedding_dim)
        output, state = self.gru(x, hidden)
        # output.shape == (batch_size, 1, dec_units)
        x = self.fc(output)
        # x.shape == (batch_size, 1, vocab_size)
        # state.shape == (num_layers, batch_size, dec_units)
        return x, state, None

    def from_pretrained(self, embedding_matrix):
        if not isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = torch.tensor(embedding_matrix).float()
        self.embedding = Embedding.from_pretrained(embedding_matrix)


class AttentionDecoder(Module):

    def __init__(self, dec_units, embedding_dim, vocab_size=10000, num_layers=1,
                 attn_layer=ConcatAttention, attn_hidden_size=64):
        super().__init__()
        self.dec_units = dec_units
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(embedding_dim + dec_units, dec_units, num_layers=num_layers, batch_first=True)
        self.attention = attn_layer(dec_units, attn_hidden_size)
        self.fc = Linear(dec_units, vocab_size)

    def forward(self, x, hidden=None, encoder_outputs=None, mask=None):
        # input.shape == (batch_size, 1)
        # encoder_outputs.shape == (batch_size, seq_len, enc_units)

        x = self.embedding(x)
        # x.shape == (batch_size, 1, embedding_dim)

        # Attention
        context_vector, attention_weights = self.attention(encoder_outputs, context=hidden, mask=mask)
        # context_vec.shape == (batch_size, enc_units)
        # attention_weights.shape == (batch_size, seq_len, 1)

        # Reshape to make concat work
        context_vector = context_vector.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat((x, context_vector), dim=-1)
        #x.shape == (batch_size, 1, embedding_dim + enc_units)

        output, state = self.gru(x, hidden)
        # output.shape == (batch_size, 1, dec_units)
        x = self.fc(output)
        # x.shape == (batch_size, 1, vocab_size)
        # state.shape == (num_layers, batch_size, dec_units)
        return x, state, attention_weights

    def from_pretrained(self, embedding_matrix):
        if not isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = torch.tensor(embedding_matrix).float()
        self.embedding = Embedding.from_pretrained(embedding_matrix)


class Seq2Seq(Module):

    def __init__(self, encoder, decoder, teacher_forcing=0., max_length=32):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing = teacher_forcing
        self.max_length = max_length

    def forward(self, source, target=None, src_mask=None):
        # source.shape == (batch_size, source_length) LONG
        # target.shape == (batch_size, target_length) LONG
        # src_mask.shape == (batch_size, source_length) BOOL: 1 == PAD; 0 == Not PAD
        batch_size, source_length = source.size()
        target_length = target.size(1) if target is not None else self.max_length

        encoder_outputs, state = self.encoder(source)
        # encoder_outputs.shape == (batch_size, source_length, enc_units)
        # state.shape == (encoder_num_layers, batch_size, enc_units)

        decoder_input = torch.zeros(batch_size, 1).long() # Assumes SOS Tok == 0
        # decoder_input.shape == (batch_size, 1) LONG
        decoder_outputs, attention_weights = [], []

        for i in range(target_length):
            decoder_output, state, attention_weight = self.decoder(decoder_input, state, encoder_outputs, src_mask)
            decoder_outputs.append(decoder_output)
            if attention_weight is not None:
                attention_weights.append(attention_weight)
            # decoder_output.shape == (batch_size, 1, target_vocab_size)
            # attention_weight.shape == (batch_size, seq_len, 1) or None


            # Update input token to decoder
            _, decoder_input = decoder_output.topk(1)
            decoder_input = decoder_input.squeeze(1).detach()
            if target is not None and random.random() < self.teacher_forcing:
                decoder_input = target[:, i].unsqueeze(1)
            # decoder_input.shape == (batch_size, 1) LONG

        decoder_outputs = torch.cat(decoder_outputs, dim=-2) # cat on seq dim
        # decoder_outputs.shape == (batch_size, target_seq_len, target_vocab_size)
        if len(attention_weights) != 0:
            attention_weights = torch.cat(attention_weights, dim=-1).transpose(1, 2) # cat on last dim
            # attention_weights.shape == (batch_size, target_seq_len, source_seq_len)
        else:
            attention_weights = None

        return decoder_outputs, attention_weights
