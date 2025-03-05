import math, wav_2_letter
from torch import nn
from model import PositionalEncoding, TransformerEncoderBlock
from audio_data import base_channels
    
class TransformerCTCEncoder(nn.Module):
    """The Transformer encoder."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, vocab_size, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.to_hidden = nn.LazyLinear(num_hiddens, bias=use_bias)
        self.to_logits = nn.LazyLinear(vocab_size, bias=use_bias)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):

        X = self.pos_encoding(self.to_hidden(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights

        return nn.functional.log_softmax(self.to_logits(X), dim = -1)
    

class comboModel(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, vocab_size, use_bias=False):
        super().__init__()
        self.encoder = TransformerCTCEncoder(num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, vocab_size, use_bias)
        self.to_channels = nn.LazyLinear(base_channels * 3, bias=use_bias)
        self.wav_2_letter = wav_2_letter.wav2letter()

    def forward(self, X, valid_lens):
        X = self.encoder(X, valid_lens)
        X = self.to_channels(X)
        X = X.permute(0, 2, 1)
        X = self.wav_2_letter(X)
        X = X.permute(0, 2, 1)

        return X