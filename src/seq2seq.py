import torch
import random
from torch.nn import (
    Module, GRU, Embedding, Linear
)


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

    def forward(self, x, hidden=None, encoder_outputs=None):
        # input.shape == (batch_size, 1)
        x = self.embedding(x)
        # x.shape == (batch_size, 1, embedding_dim)
        output, state = self.gru(x, hidden)
        # output.shape == (batch_size, 1, dec_units)
        x = self.fc(output)
        # x.shape == (batch_size, 1, vocab_size)
        # state.shape == (num_layers, batch_size, dec_units)
        return x, state

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

    def forward(self, source, target=None):
        # source.shape == (batch_size, source_length) LONG
        # target.shape == (batch_size, target_length) LONG
        batch_size, source_length = source.size()
        target_length = target.size(1) if target is not None else self.max_length

        encoder_outputs, state = self.encoder(source)
        # encoder_outputs.shape == (batch_size, source_length, enc_units)
        # state.shape == (encoder_num_layers, batch_size, enc_units)

        decoder_input = torch.zeros(batch_size, 1).long() # Assumes SOS Tok == 0
        # decoder_input.shape == (batch_size, 1) LONG
        decoder_outputs = []

        for i in range(target_length):
            decoder_output, state = self.decoder(decoder_input, state, encoder_outputs)
            decoder_outputs.append(decoder_output)
            # decoder_output.shape == (batch_size, 1, target_vocab_size)

            # Update input token to decoder
            _, decoder_input = decoder_output.topk(1)
            decoder_input = decoder_input.squeeze(1).detach()
            if target is not None and random.random() < self.teacher_forcing:
                decoder_input = target[:, i].unsqueeze(1)
            # decoder_input.shape == (batch_size, 1) LONG

        decoder_outputs = torch.cat(decoder_outputs, dim=-2) # cat on seq dim
        # decoder_outputs.shape == (batch_size, target_seq_len, target_vocab_size)

        return decoder_outputs
