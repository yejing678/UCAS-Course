import torch.nn as nn


class LSTM_LM(nn.Module):
    def __init__(self, opt, vocab_size):
        super(LSTM_LM, self).__init__()
        self.opt = opt
        self.hidden_dim = opt.hidden_dim
        self.embedding = nn.Embedding(vocab_size, opt.hidden_dim)
        self.lstm = nn.LSTM(input_size=opt.hidden_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layers, dropout=opt.dropout)
        self.dropout = nn.Dropout(opt.dropout)
        self.lm_head = nn.Linear(opt.hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        input_embeds = self.embedding(input_ids)
        lstm_output, output_hidden = self.lstm(input_embeds, hidden)
        lstm_output = self.dropout(lstm_output)
        lstm_output = lstm_output.reshape(-1, self.hidden_dim)
        lm_output = self.lm_head(lstm_output)
        return lm_output, output_hidden