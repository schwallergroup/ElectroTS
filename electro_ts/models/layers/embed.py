import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    """Inverted embedding: projects each variate's time-series into d_model space."""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [Batch, Time, Variate] -> permute to [Batch, Variate, Time]
        x = x.permute(0, 2, 1)
        if x_mark is not None:
            x = torch.cat([x, x_mark.permute(0, 2, 1)], dim=1)
        return self.dropout(self.value_embedding(x))
