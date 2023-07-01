import math
import torch
import torch.nn as nn

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape  # (vld,25,128)
        W = int(math.sqrt(WW))    # 5
        scale = data['hw0_i'][0] / data['hw0_f'][0]  # 2
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        # kernel 中心的 feature
        feat_f0_picked = feat_f0_picked = feat_f0[:, WW//2, :]  # (vld,128)
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)  # (vld,25)
        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)  # (vld,5,5)

        # compute sub-pixel coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # (vld,2)
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # (1,25,2)

        # compute std over <x, y>
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # (vld,2)
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # (vld) clamp needed for numerical stability
        
        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})  # (vld,3)

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = data['mkpts0_c']  # (vld,2), coarse 像素坐标
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]

        data.update({
            "mkpts0_f": mkpts0_f,  # fine-tune 坐标
            "mkpts1_f": mkpts1_f
        })
