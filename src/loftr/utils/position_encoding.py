import math
import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))  # (256,256,256)
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)  # (1,256,256)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)  # (1,256,256)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
            # (C//4)
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # (64,1,1)
        pe[0::4, :, :] = torch.sin(x_position * div_term)  # (64,256,256)
        pe[1::4, :, :] = torch.cos(x_position * div_term)  # (64,256,256)
        pe[2::4, :, :] = torch.sin(y_position * div_term)  # (64,256,256)
        pe[3::4, :, :] = torch.cos(y_position * div_term)  # (64,256,256)

        # 相同坐标，不同 channel，是不同的编码
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # (1,C(256),256,256)

    def forward(self, x):
        """
        Args:
            x: [N, C, H/8, W/8]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
