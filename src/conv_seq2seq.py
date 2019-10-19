import torch
from torch.nn import (
    Module, ModuleList, Sequential,
    Embedding, Linear, Dropout, Conv1d
)
import torch.nn.functional as F

class EncoderConv(Module):

    def __init__(self, in_channels=512, out_channels=1024, kernel_size=3, padding=1, dropout=0.25):
        super().__init__()
        self.conv = Conv1d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, padding=padding)
        self.dropout = Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))

    def forward(self, conv_input):
        # conv_input.shape == (batch_size, hidden_size, seq_len)
        conved = self.conv(self.dropout(conv_input))
        # conved.shape == (batch_size, 2*hidden_size, seq_len)
        conved = F.glu(conved, dim=1)
        # conved.shape == (batch_size, hidden_size, seq_len)
        conved = (conved + conv_input) * self.scale # residual
        # conved.shape == (batch_size, hidden_size, seq_len)
        return conved

class Encoder(Module):

    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, num_layers=1, kernel_size=3, dropout=0.25, max_length=100):
        super().__init__()
        assert kernel_size % 2 == 1, f"Kernel size must be odd, got {kernel_size}"
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))

        self.token_embedding = Embedding(vocab_size, embedding_dim)
        self.position_embedding = Embedding(max_length, embedding_dim)

        self.embed2hidden = Linear(embedding_dim, hidden_size)
        self.hidden2embed = Linear(hidden_size, embedding_dim)

        convs = [EncoderConv(hidden_size, 2 * hidden_size, kernel_size, (kernel_size - 1) // 2, dropout) for _ in range(num_layers)]
        self.convs = Sequential(*convs)

        self.dropout = Dropout(dropout)

    def forward(self, src):
        # src.shape = (batch_size, seq_len)
        # pos.shape = (batch_size, seq_len)
        batch_size, seq_len = src.size()
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1)

        tok_embedded = self.token_embedding(src)
        pos_embedded = self.position_embedding(pos)
        # tok_embedded.shape = pos_embedded.shape = (batch_size, seq_len, embedding_dim)

        embedded = self.dropout(tok_embedded + pos_embedded)
        # embedded.shape == (batch_size, seq_len, embedding_dim)

        conv_input = self.embed2hidden(embedded).permute(0, 2, 1)
        # conv_input.shape == (batch_size, hidden_size, seq_len)

        conved = self.convs(conv_input)
        # conved.shape == (batch_size, hidden_size, seq_len)

        conved = self.hidden2embed(conved.permute(0, 2, 1))
        # conved.shape == (batch_size, seq_len, embedding_dim)

        combined = (conved + embedded) * self.scale
        # combined.shape == (batch_size, seq_len, embedding_dim)
        return conved, combined


class Attention(Module):

    def __init__(self, embedding_dim, hidden_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.hidden2embed = Linear(hidden_size, embedding_dim)
        self.embed2hidden = Linear(embedding_dim, hidden_size)
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))

    def forward(self, embedded, conved, encoder_conved, encoder_combined):
        # embedded.shape == (batch_size, tgt_seq_len, embedding_dim)
        # conved == (batch_size, hidden_size, tgt_seq_len)
        # encoder_conved = encoder_combined == (batch_size, src_seq_len, embedding_dim)

        conved_emb = self.hidden2embed(conved.permute(0, 2, 1))
        # conved_emb.shape == (batch_size, tgt_seq_len, embedding_dim)

        combined = (conved_emb + embedded) * self.scale
        # combined == (batch_size, tgt_seq_len, embedding_dim)

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        # energy == (batch_size, tgt_seq_len, src_seq_len)

        attention = F.softmax(energy, dim=-1)
        # attention == (batch_size, tgt_seq_len, src_seq_len)

        attended_encoding = torch.matmul(attention, encoder_combined)
        # attended_encoding == (batch_size, tgt_seq_len, embedding_dim)

        attended_encoding = self.embed2hidden(attended_encoding)
        # attended_encoding == (batch_size, tgt_seq_len, hidden_dim)

        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        # attended_combined == (batch_size, hidden_dim, tgt_seq_len)
        return attention, attended_combined


class DecoderConv(Module):

    def __init__(self, kernel_size=3, dropout=0.25, PAD_token=0):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.PAD_token = PAD_token
        self.kernel_size = kernel_size
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))


    def forward(self, conv_layer, attn_layer, conv_input, embedded, encoder_conved, encoder_combined):
        conv_input = self.dropout(conv_input)
        batch_size, hidden_dim, tgt_seq_len = conv_input.size()

        padding = torch.zeros(batch_size, hidden_dim, self.kernel_size - 1).fill_(self.PAD_token)

        padded_conv_input = torch.cat((padding, conv_input), dim=2)
        # padded_conv_input = (batch_size, hidden_dim, tgt_seq_len + kernel_size - 1)

        conved = conv_layer(padded_conv_input)
        # conved = (batch_size, 2*hidden_dim, tgt_seq_len)

        conved = F.glu(conved, dim=1)
        # conved = (batch_size, hidden_dim, tgt_seq_len)

        attention, conved = attn_layer(embedded, conved, encoder_conved, encoder_combined)
        # attention = (batch_size, tgt_seq_len, src_seq_len)

        conved = (conved + conv_input) * self.scale
        # conved.shape == (batch_size, hidden_dim, tgt_seq_len)
        return conved, attention.transpose(1, 2)


class Decoder(Module):

    def __init__(self, hidden_dim=512, embedding_dim=256, vocab_size=10000, num_layers=10,
                 kernel_size=3, dropout=0.25, PAD_token=0, max_len=50):
        super().__init__()
        self.kernel_size = kernel_size
        self.PAD_token = PAD_token
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.token_embedding = Embedding(vocab_size, embedding_dim)
        self.position_embedding = Embedding(max_len, embedding_dim)

        self.embedd2hidden = Linear(embedding_dim, hidden_dim)
        self.hidden2embedd = Linear(hidden_dim, embedding_dim)

        self.attention_layer = Attention(embedding_dim, hidden_dim)
        self.decoder_conv = DecoderConv(kernel_size, dropout, PAD_token)
        self.convs = ModuleList([Conv1d(in_channels=hidden_dim, out_channels=2*hidden_dim,
                                        kernel_size=kernel_size) for _ in range(num_layers)])

        self.out = Linear(embedding_dim, vocab_size)

        self.dropout = Dropout(dropout)

    def forward(self, target, encoder_conved, encoder_combined):
        # target.shape == (batch_size, tgt_seq_len)
        # encoder_conved = encoder_combined == (batch_size, src_seq_len, embedding_dim)
        batch_size, target_seq_len = target.size()
        source_seq_len = encoder_conved.size(1)

        pos = torch.arange(0, target_seq_len).unsqueeze(0).repeat(batch_size, 1)
        tok_embedded = self.token_embedding(target)
        pos_embedded = self.position_embedding(pos)
        # pos_embedded = tok_embedded = (batch_size, tgt_seq_len, embedding_dim)

        embedded = self.dropout(tok_embedded + pos_embedded)
        # embedded.shape == (batch_size, tgt_seq_len, embedding_dim)

        conv_input = self.embedd2hidden(embedded).permute(0, 2, 1)
        # conv_input = (batch_size, hidden_size, target_seq_len)

        ############
        # Loop for convs
        for conv in self.convs:
            conv_input, attention = self.decoder_conv(conv, self.attention_layer,
                                                      conv_input, embedded,
                                                      encoder_conved, encoder_combined)

        ############
        # conved_input == (batch_size, hidden_dim, tgt_seq_len)
        # attention == (batch_size, tgt_seq_len, src_seq_len)

        conved = self.hidden2embedd(conv_input.permute(0, 2, 1))
        # conved = (batch_size, target_seq_len, embedding_dim)

        output = self.out(self.dropout(conved))
        # output == (batch_size, tgt_seq_len, target_vocab_size)
        return output, attention



class Seq2Seq(Module):

    def __init__(self, encoder, decoder, max_len=50, SOS_token=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_len = max_len
        self.SOS_token = SOS_token
        self.type = 'conv'

    def forward(self, src, tgt=None, src_mask=None):
        # src = (batch_size, src_seq_len)
        # tgt = (batch_size, tgt_seq_len)
        batch_size = src.size(0)

        encoder_conved, encoder_combined = self.encoder(src)
        # encoder_conved.shape == (batch_size, src_seq_len, embedding_dim)
        # encoder_combined.shape == (batch_size, src_seq_len, embedding_dim)

        # diverge for train/inference:
        if tgt is not None:
            decoder_outputs, attention_weights = self.decoder(tgt, encoder_conved, encoder_combined)
            # output.shape == (batch_size, tgt_seq_len, tgt_vocab_size)
            # attention == (batch_size, tgt_seq_len, src_seq_len)
        else:
            decoder_input = torch.full((batch_size, 1), self.SOS_token).long()
            decoder_outputs, attention_weights = [], []

            for i in range(self.max_len):
                decoder_output, attention_weight = self.decoder(decoder_input, encoder_conved, encoder_combined)
                # decoder_output == (batch_size, i+1, tgt_vocab_size)
                # attention_weight == (batch_size, src_seq_len, i)
                decoder_outputs.append(decoder_output[:, -1, :].unsqueeze(1))
                attention_weights.append(attention_weight[:, :, -1].unsqueeze(2))
                pred_token = decoder_output.argmax(dim=-1)[:, -1].unsqueeze(1).detach()
                # print(pred_token)
                decoder_input = torch.cat((decoder_input, pred_token), dim=-1)
                # decoder_input = (batch_size, i + 1)

            decoder_outputs = torch.cat(decoder_outputs, dim=-2)
            attention_weights = torch.cat(attention_weights, dim=-1).transpose(1, 2)
            # decoder_outputs == (batch_size, max_len, tgt_vocab_size)
            # attention_weights == (batch_size, max_len, src_seq_len)
            # print(decoder_outputs.size(), attention_weights.size())
        return decoder_outputs, attention_weights

# EOF
