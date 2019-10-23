import torch
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math
import unicodedata
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import sys


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

    def __init__(self, max_vocab_size=10000):
        self.vocab = tf.keras.preprocessing.text.Tokenizer(num_words=max_vocab_size, filters='', oov_token='<unk>')
        self.PAD_token = 0
        self.max_vocab_size = max_vocab_size

    def build_vocab(self, data):
        self.vocab.fit_on_texts(data)
        self.SOS_token = self.vocab.word_index['<sos>']
        self.EOS_token = self.vocab.word_index['<eos>']
        self.UNK_token = self.vocab.word_index['<unk>']
        self.num_words = min(self.vocab.num_words, len(self.vocab.word_index) + 1)

    def to_sequence(self, data):
        tensor = self.vocab.texts_to_sequences(data)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        return torch.tensor(tensor).long()

    def is_special_token(self, token):
        return token.item() in [self.PAD_token, self.EOS_token, self.SOS_token]

    def to_string(self, tensor, remove_special=False):
        if remove_special:
            return [ " ".join([self.vocab.index_word[idx.item()] for idx in t if not self.is_special_token(idx)]) for t in tensor]
        return [ " ".join([self.vocab.index_word[idx.item()] for idx in t if idx.item() != self.PAD_token]) for t in tensor]

    def __len__(self):
        return self.num_words


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# change to spacy?
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

def make_dataset(train_text, test_text, train_batch_size=32, test_batch_size=64, max_vocab_size=10000):
    src_train = [preprocess_sentence(t) for t in train_text[0]]
    src_test = [preprocess_sentence(t) for t in test_text[0]]
    tgt_train = [preprocess_sentence(t) for t in train_text[1]]
    tgt_test = [preprocess_sentence(t) for t in test_text[1]]
    # src_train, src_test, tgt_train, tgt_test = train_test_split(source_text, target_text, test_size=test_size)
    src_vocab, tgt_vocab = Vocab(max_vocab_size), Vocab(max_vocab_size)
    src_vocab.build_vocab(src_train); tgt_vocab.build_vocab(tgt_train)
    src_train, src_test = src_vocab.to_sequence(src_train), src_vocab.to_sequence(src_test)
    tgt_train, tgt_test = tgt_vocab.to_sequence(tgt_train), tgt_vocab.to_sequence(tgt_test)
    train_loader = DataLoader(TensorDataset(src_train, tgt_train), batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(src_test, tgt_test), batch_size=test_batch_size, shuffle=True)
    return src_vocab, tgt_vocab, train_loader, test_loader

##########################

def train(model, iterator, optimizer, criterion, clip=1, pad_tok=0):
    model.train()
    epoch_loss = 0
    for i, (src, tgt) in enumerate(tqdm(iterator, file=sys.stdout)):
        optimizer.zero_grad()
        # src.shape = (batch_size, src_seq_len)
        # tgt.shape = (batch_size, tgt_seq_len)
        src_mask = create_padding_mask(src, pad_tok)

        if model.type == 'rnn':
            output, _ = model(src, tgt, src_mask=src_mask)
            # output.shape == (batch_size, tgt_seq_len, tgt_vocab_size)
            # output = output[:, 1:, :]
            tgt = tgt[:, 1:]
            # loss = criterion(output, tgt)
        elif model.type == 'conv':
            output, _ = model(src, tgt)
            # print(output.size())
            # print(tgt.size(), tgt[:,1:].size())
            tgt = tgt[:,1:]

        loss = criterion(output, tgt)


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, pad_tok=0):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, tgt) in enumerate(tqdm(iterator, file=sys.stdout)):
            # src.shape = (batch_size, src_seq_len)
            # tgt.shape = (batch_size, tgt_seq_len)
            src_mask = create_padding_mask(src, pad_tok)

            if model.type == 'rnn':
                output, attention = model(src, None, src_mask) #turn off teacher forcing
                # output.shape == (batch_size, max_length, tgt_vocab_size)
                # print(output)
                # output = output[:, 1:, :]
                tgt = tgt[:, 1:]
            elif model.type == 'conv':
                output, attention = model(src, None) #turn off teacher forcing
                tgt = tgt[:, 1:]

            loss = criterion(output, tgt) # masked loss automatically slices for you

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def translate(sentence, model, src_vocab, tgt_vocab, pad_tok=0):
    with torch.no_grad():
        model.eval()
        if type(sentence) == str:
            sentence = [sentence]
        tokenized_sentence = [preprocess_sentence(sent) for sent in sentence]
        tensor = src_vocab.to_sequence(tokenized_sentence)
        tokenized_sent = src_vocab.to_string(tensor, remove_special=True)[0]
        mask = create_padding_mask(tensor, pad_tok)
        print(tensor)
        translation_tensor_logits, attention = model(tensor, None, mask)
        translation_tensor = torch.argmax(translation_tensor_logits, dim=-1)
        print(translation_tensor)
        translation = tgt_vocab.to_string(translation_tensor, remove_special=True)[0]
        if attention is not None:
            attention = attention.detach().squeeze(0)[:len(translation.split()),:len(tokenized_sent.split())]
        return translation, attention

def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention.detach().squeeze(0), cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence.split(), fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence.split(), fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

def load_data(lang1='en', lang2='de'):
    train = './data/train.'
    val = './data/val.'
    src_train = [line.rstrip('\n') for line in open(f"{train}{lang1}")]
    tgt_train = [line.rstrip('\n') for line in open(f"{train}{lang2}")]
    src_test = [line.rstrip('\n') for line in open(f"{val}{lang1}")]
    tgt_test = [line.rstrip('\n') for line in open(f"{val}{lang2}")]
    return (src_train, tgt_train), (src_test, tgt_test)

def load_summary(N=2000):
    src = './data/sumdata/train/train.article.txt'
    tgt = './data/sumdata/train/train.title.txt'
    with open(src) as src_file:
        src_train = [next(src_file).rstrip('\n') for _ in range(N)]
    with open(tgt) as tgt_file:
        tgt_train = [next(tgt_file).rstrip('\n') for _ in range(N)]
    return src_train, tgt_train
