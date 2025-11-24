import torch
import torch.nn as nn
import timm

class proj_head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int = 256, bottleneck_dim: int = 256, nlayers: int = 3):
        super().__init__()
        layers = []
        dim_list = [in_dim] + [hidden_dim] * (nlayers - 2) + [bottleneck_dim]
        for i in range(nlayers - 1):
            layers += [nn.Linear(dim_list[i], dim_list[i + 1], bias=False), nn.GELU(), nn.BatchNorm1d(dim_list[i + 1])]
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=1, eps=1e-8)
        x = self.last_layer(x)
        return x