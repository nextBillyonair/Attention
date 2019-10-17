import torch
from torch.nn import (
    Module, TransformerEncoderLayer, TransformerEncoder,
    Linear, Dropout, LayerNorm, Sequential, Embedding, ModuleList,
    CrossEntropyLoss
)
from torch.nn.functional import relu
import math
from .embeddings import PositionalEmbeddings


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


##### NATIVE IMPLEMENTATION
# https://www.tensorflow.org/tutorials/text/transformer

def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    dk = key.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(dk)

    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(1)
        scaled_attention_logits += (mask * -1e10)

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, value)

    return output, attention_weights



class MultiheadAttention(Module):

    def __init__(self, d_model, num_heads, hidden_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.hidden_size = d_model if hidden_size is None else hidden_size

        assert self.hidden_size % self.num_heads == 0, f"Hidden Dimension of model (hidden_size:{hidden_size}) must be divisible by num_heads ({num_heads})"

        self.depth = self.hidden_size // self.num_heads

        self.wq = Linear(d_model, self.hidden_size)
        self.wk = Linear(d_model, self.hidden_size)
        self.wv = Linear(d_model, self.hidden_size)

        self.dense = Linear(self.hidden_size, d_model)

    def split_head(self, x):
        x = x.reshape(x.size(0), -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, value, key, query, mask=None):
        # inputs.shape == (batch_size, seq_len, d_model)
        batch_size = query.size(0)

        query = self.split_head(self.wq(query))
        key = self.split_head(self.wk(key))
        value = self.split_head(self.wv(value))

        scaled_attention, attention_weights = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = scaled_attention.transpose(1, 2)

        concat_attention = scaled_attention.reshape(batch_size, -1, self.hidden_size)

        output = self.dense(concat_attention)
        # output.shape == (batch_size, seq_len_q, d_model)
        return output, attention_weights


class PointwiseFeedForward(Module):

    def __init__(self, d_model, dff):
        super().__init__()
        self.dense_1 = Linear(d_model, dff)
        self.dense_2 = Linear(dff, d_model)

    def forward(self, x):
        # IN + OUT == (batch_size, seq_len, d_model)
        return self.dense_2(relu(self.dense_1(x)))


class EncoderLayer(Module):

    def __init__(self, d_model, num_heads, dff, mha_hidden=None, rate=0.1):
        super().__init__()

        self.mha = MultiheadAttention(d_model, num_heads, hidden_size=mha_hidden)
        self.ffn = PointwiseFeedForward(d_model, dff)

        self.layernorm1 = LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = LayerNorm(d_model, eps=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def forward(self, x, mask=None):

        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        # out2.shape == (batch_size, input_seq_len, d_model)
        return out2


class DecoderLayer(Module):

    def __init__(self, d_model, num_heads, dff, mha_hidden=None, rate=0.1):
        super().__init__()

        self.mha1 = MultiheadAttention(d_model, num_heads, hidden_size=mha_hidden)
        self.mha2 = MultiheadAttention(d_model, num_heads, hidden_size=mha_hidden)

        self.ffn = PointwiseFeedForward(d_model, dff)

        self.layernorm1 = LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = LayerNorm(d_model, eps=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)
        # out3.shape == (batch_size, target_seq_len, d_model)
        return out3, attn_weights_block1, attn_weights_block2


class Encoder(Module):

    def __init__(self, num_layers, d_model, num_heads, dff,
                 mha_hidden=None,
                 input_vocab_size=10000,
                 maximum_position_encoding=5000, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = input_vocab_size
        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEmbeddings(d_model, rate, maximum_position_encoding)

        enc_layers = [EncoderLayer(d_model, num_heads, dff, mha_hidden, rate) for _ in range(num_layers)]
        self.enc_layers = ModuleList(enc_layers)


    def forward(self, x, mask=None):
        # input == (batch_size, input_seq_len) LONG
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        # x.shape == (batch_size, input_seq_len, d_model)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        # x.shape == (batch_size, input_seq_len, d_model)
        return x

    def from_pretrained(self, embedding_matrix):
        if not isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = torch.tensor(embedding_matrix).float()
        self.embedding = Embedding.from_pretrained(embedding_matrix)


class Decoder(Module):

    def __init__(self, num_layers, d_model, num_heads, dff,
                 mha_hidden=None,
                 target_vocab_size=1000,
                 maximum_position_encoding=5000, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = target_vocab_size

        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEmbeddings(d_model, rate, maximum_position_encoding)

        dec_layers = [DecoderLayer(d_model, num_heads, dff, mha_hidden, rate) for _ in range(num_layers)]
        self.dec_layers = ModuleList(dec_layers)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        attention_weights = {}

        # input == (batch_size, input_seq_len) LONG
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        # x.shape == (batch_size, target_seq_len, d_model)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output,
                                                   look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        # attention_weights['key'].shape == (batch_size, num_heads, tar_seq_len, target_seq_len)
        return x, attention_weights

    def from_pretrained(self, embedding_matrix):
        if not isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = torch.tensor(embedding_matrix).float()
        self.embedding = Embedding.from_pretrained(embedding_matrix)


class Transformer(Module):

    def __init__(self, num_layers, d_model, num_heads, dff,
                 mha_hidden=None,
                 input_vocab_size=10000, target_vocab_size=10000,
                 pe_input=5000, pe_target=5000, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, mha_hidden,
                               input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, mha_hidden,
                               target_vocab_size, pe_target, rate)

        self.final_layer = Linear(d_model, target_vocab_size)

    def forward(self, input, target, enc_padding_mask=None,
                look_ahead_mask=None, dec_padding_mask=None):
        # enc_output.shape == (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(input, enc_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(target, enc_output, look_ahead_mask, dec_padding_mask)
        # final_output.shape == (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


class MaskedCrossEntropyLoss(Module):

    def __init__(self, pad_tok=0):
        super().__init__()
        self.criterion = CrossEntropyLoss(reduction='mean', ignore_index = pad_tok)

    def forward(self, predictions, targets):
        batch_size, tgt_seq_len = targets.size()
        targets = targets.reshape(-1)
        predictions = predictions[:, :tgt_seq_len, :]
        predictions = predictions.reshape(-1, predictions.size(-1))
        return self.criterion(predictions, targets)


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
