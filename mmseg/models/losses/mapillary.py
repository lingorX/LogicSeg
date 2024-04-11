import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .asymmetric_loss import AsymmetricLossOptimized

hiera = {
    "hiera_middle":{
        "animal": [0, 2],
        "barrier": [2, 12],
        "flat": [12, 26],
        "structure": [26, 30],
        "person": [30, 32],
        "rider": [32, 35],
        "marking_continuous": [35, 38],
        "marking_discrete": [38, 55],
        "marking_only_": [55, 59],
        "nature": [59, 66],
        "object": [66, 78],
        "sign": [78, 84],
        "support": [84, 90],
        "traffic_light": [90, 96],
        "traffic_sign": [96, 105],
        "vehicle": [105, 117],
        "void": [117, 123]
    },
    "hiera_high":{
        "animal":[0,2],
        "construction":[2, 30],
        "human":[30, 35],
        "marking":[35, 59],
        "nature": [59, 66],
        "object_":[66, 117],
        "void": [117, 123]
    }
}

def prepare_targets(targets):
    b, h, w = targets.shape
    targets_middle = torch.ones((b, h, w), dtype=targets.dtype, device=targets.device)*255
    indices_middle = []
    targets_high = torch.ones((b, h, w), dtype=targets.dtype, device=targets.device)*255
    indices_high = []
    for index, middle in enumerate(hiera["hiera_middle"].keys()):
        indices = hiera["hiera_middle"][middle]
        for ii in range(indices[0], indices[1]):
            targets_middle[targets==ii] = index
        indices_middle.append(indices)
    targets_middle[targets_middle==255] = 16
        
    for index, high in enumerate(hiera["hiera_high"].keys()):
        indices = hiera["hiera_high"][high]
        for ii in range(indices[0], indices[1]):
            targets_high[targets==ii] = index
        indices_high.append(indices)
    targets_high[targets_high==255] = 6
     
    return targets, targets_middle, targets_high, indices_middle, indices_high


def loss_bce(predictions, targets, targets_middle, targets_top, num_classes, eps=1e-8, gamma=2):
    
    void_indices = (targets==255)
    targets[void_indices]=123
    targets = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2)
    targets_middle = F.one_hot(targets_middle, num_classes = 17).permute(0,3,1,2)
    targets_top = F.one_hot(targets_top, num_classes = 7).permute(0,3,1,2)
    
    targets = torch.cat([targets, targets_middle, targets_top], dim=1)
    
    bce = F.binary_cross_entropy_with_logits(predictions, targets.float())
    loss = bce*10
    
    return loss
    
def losses_bce_focal(predictions, targets, targets_middle, targets_top, num_classes, eps=1e-8, gamma=2):
    b, _, h, w = predictions.shape
    predictions = torch.sigmoid(predictions.float())
    void_indices = (targets==255)
    targets[void_indices]=123
    targets = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2)
    targets_middle = F.one_hot(targets_middle, num_classes = 17).permute(0,3,1,2)
    targets_top = F.one_hot(targets_top, num_classes = 7).permute(0,3,1,2)
           
    targets = torch.cat([targets, targets_middle, targets_top], dim=1)
    valid_indices = (~void_indices).unsqueeze(1)
    
    loss = ((-targets*torch.pow((1.0-predictions),gamma)*torch.log(predictions+eps)
             -(1.0-targets)*torch.pow(predictions, gamma)*torch.log(1.0-predictions+eps))
             *valid_indices).sum()/valid_indices.sum()

    return loss

def losses_bce_asl(predictions, targets, targets_middle, targets_top, num_classes, asl):
    predictions = predictions.float()
    void_indices = (targets==255)
    targets[void_indices]=123
    targets = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2)
    targets_middle = F.one_hot(targets_middle, num_classes = 17).permute(0,3,1,2)
    targets_top = F.one_hot(targets_top, num_classes = 7).permute(0,3,1,2)
    
    targets = torch.cat([targets, targets_middle, targets_top], dim=1)
    valid_indices = (~void_indices).unsqueeze(1)
   
    loss = asl(predictions, targets, valid_indices)

    return loss

@LOSSES.register_module()
class HieraLossMapillary(nn.Module):

    def __init__(self,
                 num_classes,
                 use_sigmoid=False,
                 loss_weight=1.0,
                 loss_name='hieralossmapillary'):
        super(HieraLossMapillary, self).__init__()
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
#         loss = loss_bce(cls_score, targets, targets_middle, targets_top, self.num_classes)
        loss = losses_bce_asl(cls_score, targets, targets_middle, targets_top, self.num_classes, self.asl)
#         loss = losses_bce_focal(cls_score, targets, targets_middle, targets_top, self.num_classes)
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