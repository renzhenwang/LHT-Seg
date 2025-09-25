import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss

from mmseg.registry import MODELS


hiera_map = [0,0,1,1,1,2,2,2,3,3,4,5,5,6,6,6,6,6,6]
hiera_index = [[0,2],[2,5],[5,8],[8,10],[10,11],[11,13],[13,19]]

hiera = {
    "hiera_high": {
        "flat": [0,2],
        "construction": [2, 5],
        "object": [5, 8],
        "nature": [8, 10],
        "sky": [10, 11],
        "human": [11,13],
        "vehicle": [13,19]
    }
}

def build_transition_matrix(hiera_map, num_high=7, num_fine=19):
    T = torch.zeros((num_high, num_fine))
    for fine_class, high_class in enumerate(hiera_map):
        T[high_class, fine_class] = 1.0
    return T

def get_high_level_preds(p_fine, T):
    """
    Compute high-level predictions
    p_fine: [B, C_fine, H, W]
    T: [C_high, C_fine]
    return: [B, C_high, H, W]
    """
    B, C, H, W = p_fine.shape
    T = T.to(p_fine.device)
    p_fine_flat = p_fine.permute(0, 2, 3, 1).contiguous().reshape(-1, C)  # [B*H*W, C_fine]
    p_high_flat = p_fine_flat @ T.T  # [B*H*W, C_high]
    p_high = p_high_flat.view(B, H, W, T.shape[0]).permute(0, 3, 1, 2).contiguous()
    return p_high

def prepare_targets(targets):
    """
    Generate high-level labels
    """
    b, h, w = targets.shape
    targets_high = torch.ones((b, h, w), dtype=targets.dtype, device=targets.device) * 255
    indices_high = []
    for index, high in enumerate(hiera["hiera_high"].keys()):
        indices = hiera["hiera_high"][high]
        for ii in range(indices[0], indices[1]):
            targets_high[targets == ii] = index
        indices_high.append(indices)
    return targets, targets_high, indices_high

@MODELS.register_module()
class LabelHierarchyTransitionLoss(nn.Module):
    def __init__(self, num_classes=19, num_high=7, embed_dim=64, loss_weight=1.0, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_high = num_high
        self.loss_weight = loss_weight

        self.ce = CrossEntropyLoss()
        self.register_buffer('T_fixed', build_transition_matrix(hiera_map, num_high, num_classes))

        self.scale = nn.Parameter(torch.tensor(1.0))

        self.T_dynamic = nn.Parameter(torch.zeros(num_high, num_classes))
        nn.init.xavier_uniform_(self.T_dynamic)

        self.loss_name = 'loss_decode'

    def forward(self, cls_score, label, ignore_index=-100, **kwargs):

        ce_fine = self.ce(cls_score, label, ignore_index=ignore_index)
        p_fine = torch.softmax(cls_score, dim=1)

        T_final = self.scale * self.T_fixed + self.T_dynamic
        T_final = F.softmax(T_final, dim=0)

        p_high = get_high_level_preds(p_fine, T_final)
        _, targets_high, _ = prepare_targets(label)

        ce_high = F.nll_loss(torch.log(p_high + 1e-8), targets_high, ignore_index=ignore_index)

        loss = ce_fine + ce_high

        # confusion loss
        C_high, C_fine = T_final.shape
        uniform = T_final.new_full((C_high, C_fine), 1.0 / C_high)
        conf_reg = (T_final * (T_final.log() - uniform.log())).sum(dim=0).mean()
        
        loss += 0.001 * conf_reg

        return self.loss_weight * loss
