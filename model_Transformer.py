import math
import torch
from torch import nn

TARGET_COLS = [
    "OD", "CarbonContents", "Ethanol", "Acetoin", "BDO",
    "Succinate", "Lactate", "Formate", "Acetate",
]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Input must be 3D: [B, L, D]")
        l = x.shape[1]
        return x + self.pe[:l].unsqueeze(0)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.2,
        ff_dim: int = 256,
        pre_ln: bool = True,
        out_activation: str = "relu",
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        try:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=pre_ln,
            )
            self.batch_first = True
        except TypeError:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
            )
            self.batch_first = False

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, len(TARGET_COLS))

        if out_activation == "relu":
            self.out_act = nn.ReLU()
        elif out_activation == "softplus":
            self.out_act = nn.Softplus(beta=1.0, threshold=20.0)
        elif out_activation == "none":
            self.out_act = nn.Identity()
        else:
            raise ValueError("out_activation must be 'relu', 'softplus', or 'none'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.input_proj(x))
        if self.batch_first:
            h = self.encoder(h)
            last = h[:, -1]
        else:
            h = self.encoder(h.permute(1, 0, 2))
            last = h[-1]
        out = self.head(self.norm(last))
        return self.out_act(out)