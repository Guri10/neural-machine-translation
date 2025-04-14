import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.embedding(src)  # [batch_size, src_len, emb_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [batch_size, src_len, hidden_dim]
        return outputs, hidden, cell
