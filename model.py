import math, torch, inspect, os
from torch import nn

class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of qkv: (batch_size, no. of queries, num_hiddens)
    # Shape of valid_lens: (batch_size * num_heads) or None
    def forward(self, queries, keys, values, valid_lens=None, causal: bool = False):
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])

        if valid_lens is not None: # all paths except decoder attention 1 predict path
            # Build a mask initialized as all False
            mask = torch.zeros_like(scores, dtype=torch.bool)
            if causal: # decoder attention 1 path
                seq_len = scores.size(-1)
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1)
                mask |= causal_mask.unsqueeze(0)

            # (batch_size * num_heads, num_queries, num_keys)
            pad_mask = torch.arange(scores.size(-1), device=scores.device)[None, :] >= valid_lens[:, None]
            pad_mask = pad_mask.unsqueeze(1).expand(-1, scores.size(1), -1)
            mask |= pad_mask

            # Apply combined mask
            scores = scores.masked_fill(mask, float('-inf'))

        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values) # plt.imshow(self.attention_weights[0].cpu().detach().numpy(), cmap = 'hot')

class MultiHeadAttention(nn.Module):  
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout=dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens, causal: bool = False):
        # Shape of queries, keys, or values: (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:(batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None: # all paths except decoder attention 1 predict path
            # On axis 0, copy the first item (scalar or vector) for num_heads times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens, causal)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
    
    def transpose_qkv(self, X):
        # Shape of input X: (batch_size, no. of queries or key-value pairs, num_hiddens). Shape of output X: (batch_size, no. of queries or key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):# Shape of input: (batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads) -> Shape of output X: (batch_size, no. of queries or key-value pairs, num_hiddens)
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    
class SimpleMultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.ModuleList([nn.LazyLinear(math.floor(num_hiddens / num_heads), bias=bias) for _ in range(self.num_heads)])
        self.W_k = nn.ModuleList([nn.LazyLinear(math.floor(num_hiddens / num_heads), bias=bias) for _ in range(self.num_heads)])
        self.W_v = nn.ModuleList([nn.LazyLinear(math.floor(num_hiddens / num_heads), bias=bias) for _ in range(self.num_heads)])
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)


    def forward(self, queries, keys, values, valid_lens):
        output = torch.empty(queries.shape[0], queries.shape[1], 0)
        for i in range(self.num_heads):
            output = torch.cat((output, self.attention(self.W_q[i](queries), self.W_k[i](keys), self.W_v[i](values), valid_lens)), dim = 2)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        return self.W_o(output)

class PositionalEncoding(nn.Module):  
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        return self.dropout(X + self.P[:, :X.shape[1], :].to(X.device)) # plt.imshow(self.P[:, :X.shape[1], :].squeeze().cpu().detach().numpy(), cmap = 'hot')

class PositionWiseFFN(nn.Module):  
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
class AddNorm(nn.Module):  
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerDecoderBlock(nn.Module):
    # The i-th block in the Transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens, dec_valid_lens = state[0], state[1], state[2]
        # During training, all the tokens of any output sequence are processed at the same time, so state[2][self.i] is None as initialized. When decoding any output sequence token by token during prediction, state[2][self.i] contains representations of the decoded output at the i-th block up to the current time step
        if state[3][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[3][self.i], X), dim=1) #append preds
        state[3][self.i] = key_values
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens, True)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, dec_valid_lens=None):
        return [enc_outputs, enc_valid_lens, dec_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens)) # inputs to embeddings
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights # Decoder self-attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights # Encoder-decoder attention weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
  
class TransformerEncoderBlock(nn.Module):  
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(nn.Module):  
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding values are multiplied by the square root of the embedding dimension to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
    
class EncoderDecoder(nn.Module):  
    """The base class for the encoder--decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid, dec_valid):
        enc_all_outputs = self.encoder(enc_X, enc_valid)
        dec_state = self.decoder.init_state(enc_all_outputs, enc_valid, dec_valid)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]
    
    def my_predict_step(self, src, src_valid_len, bos_id, tgt_max_len, save_attention_weights=False):
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len) # [enc_outputs, enc_valid_lens, [None] * self.num_blks] -> [enc_outputs, enc_valid_lens, decoder_block_states]
        attention_weights = []
        outputs = [torch.full((src.shape[0], 1), bos_id)] # bos tokens
        for _ in range(tgt_max_len):
            Y, dec_state = self.decoder(outputs[-1], dec_state) # latest predictions
            outputs.append(torch.argmax(Y, 2)) # append predictions
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return torch.concat(outputs[1:], 1), attention_weights

    def my_beam_search_predict_step(self, src, src_valid_len, bos_id, tgt_max_len, beam_size=4, length_penalty=0.6):
        # Step 1: Encode source
        enc_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_outputs, src_valid_len)

        # Step 2: Initialize beam
        beams = [([bos_id], 0.0, dec_state)]  # (tokens, log_prob, state)

        # Step 3: Main loop -> For all current beams, take n top hypotheses, finally keep top n beams
        for _ in range(tgt_max_len):
            new_beams = []

            for tokens, log_prob, state in beams:
                if state[2][0] != None:
                    state = state[:3] + [[layer.detach() for layer in state[2]]]
                prev_input = torch.tensor(tokens[-1]).reshape(1, 1)
                Y, new_state = self.decoder(prev_input, state)  # shape: [1, 1, vocab_size]

                probs = torch.log_softmax(Y.squeeze(1), dim=-1)  # [1, vocab_size]
                topk_probs, topk_ids = probs.topk(beam_size)  # [1, beam_size]

                for i in range(beam_size):
                    new_token = topk_ids[0, i].item()
                    new_log_prob = log_prob + topk_probs[0, i].item()
                    new_beams.append((
                        tokens + [new_token],
                        new_log_prob,
                        new_state
                    ))

            beams = sorted(new_beams, key=lambda x: x[1] / ((len(x[0]) ** length_penalty)), reverse=True)[:beam_size]

        # Step 4: Return best sequence
        best_tokens, best_log_prob, _ = max(beams, key=lambda x: x[1] / ((len(x[0]) ** length_penalty)))
        return torch.tensor(best_tokens[1:]).unsqueeze(0)  # remove BOS
    
    def loadDict(self, modelName: str, suffix: str = '_bestBleu'):
        checkpointPath = 'checkpoints\\' + modelName + suffix + '.pth'
        if os.path.exists(checkpointPath):
            state = torch.load(checkpointPath)
            self.load_state_dict(state['model_state'])
        else:
            raise FileNotFoundError("No model state found")
    
class HyperParameters:

    def save_hyperparameters(self, ignore=[]):
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)
    
class Seq2Seq(EncoderDecoder, HyperParameters):  
    """The RNN encoder--decoder for sequence to sequence learning."""
    def __init__(self, encoder, decoder, tgt_pad, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        # Adam optimizer is used here
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def load_weights(self, modelName):
        self.load_state_dict(torch.load(f'{os.getcwd()}//models//{modelName}'))