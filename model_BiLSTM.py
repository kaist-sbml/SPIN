import torch
from torch import nn

TARGET_COLS = ["OD", "CarbonContents", "Ethanol", "Acetoin", "BDO"]

class BiLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout: float = 0.2,
        out_activation: str = "relu",
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1 * 2,
            hidden_size=hidden_dim2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim2 * 2, 128)
        self.fc2 = nn.Linear(128, len(TARGET_COLS))
        if out_activation == "relu":
            self.out_act = nn.ReLU()
        elif out_activation == "softplus":
            self.out_act = nn.Softplus(beta=1.0, threshold=20.0)
        elif out_activation == "none":
            self.out_act = nn.Identity()
        else:
            raise ValueError("out_activation must be 'relu', 'softplus', or 'none'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1, _ = self.lstm1(x)
        h1 = self.drop1(h1)
        
        h2, (hn, cn) = self.lstm2(h1)
        last_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        
        h2_dropped = self.drop2(last_hidden)

        out = torch.relu(self.fc1(h2_dropped))
        out = self.fc2(out)
        
        return self.out_act(out)
