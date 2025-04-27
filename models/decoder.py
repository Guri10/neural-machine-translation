import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Attention

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout, attention):
        super(Decoder, self).__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim + hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size] (current word index)
        input = input.unsqueeze(1)  # [batch_size, 1]

        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]

        # Get attention weights → [batch_size, src_len]
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, src_len]

        # Compute context vector = weighted sum of encoder outputs
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_dim]

        # Concatenate context with embedding
        lstm_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + hidden_dim]

        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))  # outputs: [batch_size, 1, hidden_dim]

        # Combine LSTM output and context
        combined = torch.cat((outputs.squeeze(1), context.squeeze(1)), dim=1)  # [batch_size, hidden_dim*2]

        prediction = self.fc_out(combined)  # [batch_size, output_dim]

        return prediction, hidden, cell, attn_weights.squeeze(1)
