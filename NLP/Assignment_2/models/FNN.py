import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNNLM(nn.Module):
    def __init__(self, opt, vocab_size):
        super(FeedForwardNNLM, self).__init__()
        self.opt = opt
        self.embeddings = nn.Embedding(vocab_size, opt.embedding_dim)
        self.e2h = nn.Linear(opt.window_size * opt.embedding_dim, opt.hidden_dim)
        self.h2o = nn.Linear(opt.hidden_dim, vocab_size)
        self.activate = torch.tanh

    def forward(self, inputs):
        embeds = self.embeddings(inputs).reshape((inputs.shape[0], -1))
        hidden = self.activate(self.e2h(embeds))
        output = self.h2o(hidden)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs
