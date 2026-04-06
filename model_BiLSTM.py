import tensorflow as tf
from tensorflow.keras import layers, Model

TARGET_COLS = ["OD", "CarbonContents", "Ethanol", "Acetoin", "BDO"]

class BiLSTMRegressor(Model):
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout: float = 0.2,
        out_activation: str = "relu",
        **kwargs
    ):
        super(BiLSTMRegressor, self).__init__(**kwargs)
        self.lstm1 = layers.Bidirectional(
            layers.LSTM(hidden_dim1, return_sequences=True),
            input_shape=(None, input_dim)
        )
        self.drop1 = layers.Dropout(dropout)
        self.lstm2 = layers.Bidirectional(
            layers.LSTM(hidden_dim2, return_sequences=False)
        )
        self.drop2 = layers.Dropout(dropout)
        self.fc1 = layers.Dense(128, activation="relu")
        self.fc2 = layers.Dense(len(TARGET_COLS))
        if out_activation == "relu":
            self.out_act = layers.Activation("relu")
        elif out_activation == "softplus":
            self.out_act = layers.Activation("softplus")
        else:
            self.out_act = layers.Activation("linear")

    def call(self, x):
        h = self.lstm1(x)
        h = self.drop1(h)
        h = self.lstm2(h)
        h = self.drop2(h)
        h = self.fc1(h)
        out = self.fc2(h)
        return self.out_act(out)
