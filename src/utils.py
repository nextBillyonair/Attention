import torch
import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x.pow(3))))

def create_padding_mask(x, pad_tok = 0.):
    return (x == pad_tok).float()

def create_look_ahead_mask(x):
    return torch.triu(torch.ones_like(x), diagonal=1).float()

def create_masks(input, target):
    enc_padding_mask = create_padding_mask(input)
    dec_padding_mask = create_padding_mask(input)

    look_ahead_mask = create_look_ahead_mask(target)
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
