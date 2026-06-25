import torch
import torch.nn as nn

from prob_models.pie_model import MultiHeadSelfAttention


class UncertaintyModuleImage(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()
        self.attention = MultiHeadSelfAttention(1, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_in, d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)
        fc_out = self.fc2(out)
        out = self.fc(residual) + fc_out
        return {"logsigma": out, "attention": attn}
