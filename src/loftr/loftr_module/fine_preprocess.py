import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']  # True
        self.W = self.config['fine_window_size']             # 5

        d_model_c = self.config['coarse']['d_model']  # 256
        d_model_f = self.config['fine']['d_model']    # 128
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W                                     # 5
        stride = data['hw0_f'][0] // data['hw0_c'][0]  # 4

        data.update({'W': W})
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # 1. unfold(crop) all local windows
        # (N,128,H/2,W/2) 提取 kernel
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
        # (N,128*5*5,HW/64)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        # (N,HW/64,25,128)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        # (N,HW/64,25,128)

        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # (vld,25,128)
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]  # (vld,25,128)

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            # down_proj( (2vld,256) ) -> (2vld,128)
            feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
                                                   feat_c1[data['b_ids'], data['j_ids']]], 0))
            # merge_feat( (2vld,25,256) ) -> (2vld,25,128)
            feat_cf_win = self.merge_feat(torch.cat([
                torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # (2vld,25,128)
                repeat(feat_c_win, 'n c -> n ww c', ww=W**2),    # (2vld,25,128)
            ], -1))
            # 切成两块
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)
            # (vld,25,128)

        return feat_f0_unfold, feat_f1_unfold
