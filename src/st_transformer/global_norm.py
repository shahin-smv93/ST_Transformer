import torch
import torch.nn as nn

class GlobalNorm(nn.Module):
    def __init__(self, global_mean, global_std, eps=1e-5, affine=True):
        super().__init__()
        # convert to tensors and reshape to (1, 1, num_features)
        global_mean = torch.tensor(global_mean, dtype=torch.float32).view(1, 1, -1)
        global_std = torch.tensor(global_std, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("global_mean", global_mean)
        self.register_buffer("global_std", global_std)
        self.affine = affine
        self.eps = eps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones_like(global_mean))
            self.affine_bias = nn.Parameter(torch.zeros_like(global_mean))
        else:
            self.affine_weight = None
            self.affine_bias = None

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _normalize(self, x):
        out = (x - self.global_mean) / self.global_std
        if self.affine:
            out = out * self.affine_weight + self.affine_bias
        return out

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        out = x * self.global_std + self.global_mean
        return out