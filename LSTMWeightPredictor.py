import torch.nn as nn

class LSTMWeightPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=2, output_dim=None):
        super(LSTMWeightPredictor, self).__init__()

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output Layer (Linear)
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if output_dim
            else nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)  # LSTM output, hidden state, and cell state

        # We take the output from the last time step (last token in sequence)
        out = lstm_out[:, -1, :]  # Shape: (batch, hidden_dim)
        return self.output_proj(out)  # Predict next weights (output_dim)