import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .asymmetric_loss import AsymmetricLossOptimized

hiera = {
    "hiera_middle":{
        "flat": [0,9],
        "strcuture": [9, 25],
        "barrier": [25,29],
        "sign": [29,35],
        "furniture": [35,61],
        "vehicle": [61,71],
        "support": [71,74],
        "device": [74,97],
        "container": [97,112],
        "fabric": [112,119],
        "other_objects":[119,131],
        "living things":[131,133],
        "plant":[133,137],
        "nature":[137,150]
    },
    "hiera_high":{
        "Construction":[0,29],
        "Object":[29,133],
        "Nature":[133,150]
    }
}

def prepare_targets(targets):
    b, h, w = targets.shape
    targets_middle = torch.zeros((b, h, w), dtype=targets.dtype, device=targets.device)
    indices_middle = []
    targets_high = torch.zeros((b, h, w), dtype=targets.dtype, device=targets.device)
    indices_high = []
    for index, middle in enumerate(hiera["hiera_middle"].keys()):
        indices = hiera["hiera_middle"][middle]
        for ii in range(indices[0], indices[1]):
            targets_middle[targets==ii] = index
        indices_middle.append(indices)
        
    for index, high in enumerate(hiera["hiera_high"].keys()):
        indices = hiera["hiera_high"][high]
        for ii in range(indices[0], indices[1]):
            targets_high[targets==ii] = index
        indices_high.append(indices)
     
    return targets, targets_middle, targets_high, indices_middle, indices_high


def loss_bce(predictions, targets):
    bce = F.binary_cross_entropy_with_logits(predictions, targets.float())
    loss = bce*10
    
    return loss
    
def losses_bce_focal(predictions, targets, valid_indices, eps=1e-8, gamma=2):
    predictions = torch.sigmoid(predictions.float())
    loss = ((-targets*torch.pow((1.0-predictions),gamma)*torch.log(predictions+eps)
             -(1.0-targets)*torch.pow(predictions, gamma)*torch.log(1.0-predictions+eps))
             *valid_indices).sum()/valid_indices.sum()

    return loss

def losses_bce_asl(predictions, targets, valid_indices, asl):
    predictions = predictions.float()
    loss = asl(predictions, targets, valid_indices)

    return loss

@LOSSES.register_module()
class HieraLossAde20k(nn.Module):

    def __init__(self,
                 num_classes,
                 use_sigmoid=False,
                 loss_weight=1.0,
                 loss_name='hieralossade20k'):
        super(HieraLossAde20k, self).__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.asl = AsymmetricLossOptimized()

    def forward(self,
                cls_score,
                label,
                weight=None,
                **kwargs):
        """Forward function."""
        
        targets, targets_middle, targets_top, indices_middle, indices_top = prepare_targets(label)
        
        # build hierarchy GT
        void_indices = (targets==255)
        targets[void_indices]=0
        targets = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2)
        targets_middle = F.one_hot(targets_middle, num_classes = 14).permute(0,3,1,2)
        targets_top = F.one_hot(targets_top, num_classes = 3).permute(0,3,1,2)
        targets = torch.cat([targets, targets_middle, targets_top], dim=1)
        valid_indices = (~void_indices).unsqueeze(1)
    
        loss = losses_bce_asl(cls_score, targets, valid_indices, self.asl)
#         loss = losses_bce_focal(cls_score, targets, valid_indices)
#         e_rule = loss_e(cls_score_ori, self.num_classes)
#         c_rule = loss_c(cls_score_ori, self.num_classes, indices_top)
#         d_rule = loss_d(cls_score_ori, self.num_classes, indices_top)
        
#         if step<20000:
#             factor=0
#         elif step<50000:
#             factor = float(step-20000)/30000.0
#         else:
#             factor = 1.0
        
        return loss*self.loss_weight
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name