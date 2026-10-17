import torch
from torch import nn

class NextWordModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=96, hidden_dim=128, kind="rnn", dropout=0.2):
        super().__init__()
        self.kind = kind.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        recurrent = {"rnn": nn.RNN, "birnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.kind]
        self.bidirectional = self.kind == "birnn"
        self.rnn = recurrent(embedding_dim, hidden_dim, batch_first=True,
                             bidirectional=self.bidirectional,
                             dropout=0.0, num_layers=1)
        out_dim = hidden_dim * (2 if self.bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, vocab_size)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, _ = self.rnn(embedded)
        if self.bidirectional:
            # Forward summary at the final token + backward summary at the first token.
            hidden = output.size(-1) // 2
            context = torch.cat((output[:, -1, :hidden], output[:, 0, hidden:]), dim=1)
        else:
            context = output[:, -1, :]
        return self.fc(self.dropout(context))
