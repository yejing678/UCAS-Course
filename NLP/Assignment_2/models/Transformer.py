import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt=opt
        self.dropout = nn.Dropout(p=opt.dropout)


        position = torch.arange(opt.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, opt.hidden_dim, 2) * (-math.log(10000.0) / opt.hidden_dim))
        pe = torch.zeros(opt.max_len, 1, opt.hidden_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class Transformer_LM(nn.Module):

    def __init__(self,opt, vocab_size: int):
        super().__init__()
        self.opt=opt
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(opt.hidden_dim, opt.dropout)
        encoder_layers = TransformerEncoderLayer(opt.hidden_dim, opt.num_head, opt.feedforward_dim, opt.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, opt.num_layers)
        self.embedding = nn.Embedding(vocab_size, opt.hidden_dim)
        self.hidden_dim = opt.hidden_dim
        self.decoder = nn.Linear(opt.hidden_dim, vocab_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.embedding(src) * math.sqrt(self.hidden_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.dropout(output)
        output = self.decoder(output)
        return output
