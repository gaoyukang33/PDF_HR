import torch
import torch.nn as nn

class PoseDistanceMLP(nn.Module):
    def __init__(self, in_dim=29, hidden=(256, 256), dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        # softplus head guarantees non-negative predictions
        self.head = nn.Sequential(
            nn.Linear(prev, 1),
            nn.Softplus(beta=1.0, threshold=20.0)
        )

        # Kaiming init for ReLU stack
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.0)
            nn.init.zeros_(m.bias)

    def forward(self, x):            # x: (B, 29)
        h = self.backbone(x)
        y = self.head(h)             # (B, 1)
        return y.squeeze(-1)         # (B,)