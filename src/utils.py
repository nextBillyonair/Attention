import torch
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math
import unicodedata
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x.pow(3))))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


################################

def create_padding_mask(x, pad_tok = 0.):
    return (x == pad_tok).float()

def create_look_ahead_mask(x):
    return torch.triu(torch.ones_like(x), diagonal=1).float()

def create_masks(input, target, pad_tok = 0.):
    enc_padding_mask = create_padding_mask(input, pad_tok)
    dec_padding_mask = create_padding_mask(input, pad_tok)

    look_ahead_mask = create_look_ahead_mask(target)
    dec_target_padding_mask = create_padding_mask(target, pad_tok)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

###############################

# Tokens

class Vocab:

    def __init__(self):
        self.vocab = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
        self.PAD_token = 0

    def build_vocab(self, data):
        self.vocab.fit_on_texts(data)

    def to_sequence(self, data):
        tensor = self.vocab.texts_to_sequences(data)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        return torch.tensor(tensor).long()

    def to_string(self, tensor):
        text = [ " ".join([self.vocab.index_word[idx.item()] for idx in t if idx != 0]) for t in tensor]
        return text


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z,, 0-9, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z0-9$\-?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<sos> ' + w + ' <eos>'
    return w

def make_dataset(source_text, target_text, test_size=0.2, batch_size=32):
    source_text = [preprocess_sentence(t) for t in source_text]
    target_text = [preprocess_sentence(t) for t in target_text]
    src_train, src_test, tgt_train, tgt_test = train_test_split(source_text, target_text, test_size=test_size)
    src_vocab, tgt_vocab = Vocab(), Vocab()
    src_vocab.build_vocab(src_train); tgt_vocab.build_vocab(tgt_train)
    src_train, src_test = src_vocab.to_sequence(src_train), src_vocab.to_sequence(src_test)
    tgt_train, tgt_test = tgt_vocab.to_sequence(tgt_train), tgt_vocab.to_sequence(tgt_test)
    train_loader = DataLoader(TensorDataset(src_train, tgt_train))
    test_loader = DataLoader(TensorDataset(src_test, tgt_test))
    return src_vocab, tgt_vocab, train_loader, test_loader

##########################

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, tgt) in enumerate(iterator):
        optimizer.zero_grad()

        src_mask = create_padding_mask(src)

        output, attention = model(src, trg, src_mask=src_mask)
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]

        output = output[:, 1:].view(-1, output.shape[-1])
        trg = trg[:, 1:].view(-1)
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, tgt) in enumerate(iterator):

            src_mask = create_padding_mask(src)

            output, attention = model(src, None, src_mask) #turn off teacher forcing
            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output[:, 1:].view(-1, output.shape[-1])
            trg = trg[:, 1:].view(-1)

            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def translate_sentence(sentence, model, src_vocab, tgt_vocab):
    model.eval()
    tokenized_sentence = [preprocess_sentence(sentence)]
    tensor = src_vocab.to_sequence(tokenized_sentence)
    mask = create_padding_mask(tensor)
    translation_tensor_logits, attention = model(tensor, None, mask)
    translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
    translation = tgt_vocab.to_string(translation_tensor)
    return translation, attention

def display_attention(sentence, translation, attention):
    sentence = preprocess_sentence(sentence).split()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+sentence,
                       rotation=45)
    ax.set_yticklabels(['']+translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
