import torch
import torch.nn as nn
import torch.nn.functional as F

from electro_ts.models.layers.embed import DataEmbedding_inverted
from electro_ts.models.layers.transformer_enc_dec import Encoder, EncoderLayer


class STARModule(nn.Module):
    """STar Aggregate-Redistribute: O(N) inter-variate mixing via stochastic pooling."""

    def __init__(self, d_series: int, d_core: int):
        super().__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, x, *args, **kwargs):
        batch_size, channels, _ = x.shape

        combined = F.gelu(self.gen1(x))
        combined = self.gen2(combined)

        if self.training:
            ratio = F.softmax(combined, dim=1).permute(0, 2, 1).reshape(-1, channels)
            indices = torch.multinomial(ratio, 1).view(batch_size, -1, 1).permute(0, 2, 1)
            core = torch.gather(combined, 1, indices).repeat(1, channels, 1)
        else:
            weight = F.softmax(combined, dim=1)
            core = torch.sum(combined * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        out = F.gelu(self.gen3(torch.cat([x, core], dim=-1)))
        out = self.gen4(out)
        return out, None


class Model(nn.Module):
    """ElectroTS: STAR-based encoder for electrochemical time series forecasting."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.encoder = Encoder([
            EncoderLayer(
                STARModule(configs.d_model, configs.d_core),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.e_layers)
        ])
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out[:, -self.pred_len:, :]
