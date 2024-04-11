import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .asymmetric_loss import AsymmetricLossOptimized
      
hiera = {
    "hiera_high":{
        "flat": [0, 2],
        "construction": [2, 5],
        "object": [5, 8],
        "nature": [8, 10],
        "sky": [10, 11],
        "human": [11, 13],
        "vehicle": [13, 19]
    }
}

def prepare_targets(targets):
    b, h, w = targets.shape
    targets_high = torch.zeros((b, h, w), dtype=targets.dtype, device=targets.device)
    indices_high = []
    for index, high in enumerate(hiera["hiera_high"].keys()):
        indices = hiera["hiera_high"][high]
        for ii in range(indices[0], indices[1]):
            targets_high[targets==ii] = index
        indices_high.append(indices)
     
    return targets, targets_high, indices_high


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


def loss_e(predictions, num_classes, p=5.0):
    b, _, h, w = predictions.shape
    predictions = torch.sigmoid(predictions.float())
    
    MCMA = predictions[:,:-7,:,:].permute(0,2,3,1).flatten(0,2) # B*H*W, num_class
    MCMB = predictions[:,-7:,:,:].permute(0,2,3,1).flatten(0,2) # B*H*W, 7
    
    # filter high confidence pixels
    easy_A_pos = (MCMA>0.7).sum(-1)
    easy_A_neg = (MCMA<0.3).sum(-1)
    hard_A = 1 - torch.logical_and(easy_A_pos==1, easy_A_neg==num_classes-1).float()
    new_MCMA = MCMA[hard_A>0].unsqueeze(-1) # num_hard, num_class, 1
    easy_B_pos = (MCMB>0.7).sum(-1)
    easy_B_neg = (MCMB<0.3).sum(-1)
    hard_B = 1 - torch.logical_and(easy_B_pos==1, easy_B_neg==20).float()
    new_MCMB = MCMB[hard_B>0].unsqueeze(-1) # num_hard, 7, 1
    
    mask_A = (1 - torch.eye(num_classes))[None, :, :].cuda()
    mask_B = (1 - torch.eye(7))[None, :, :].cuda()
    # predicates: not (x and y)
    predicate_A = (new_MCMA@(new_MCMA.transpose(1,2)))*mask_A # num_hard, num_class, num_class
    predicate_B = (new_MCMB@(new_MCMB.transpose(1,2)))*mask_B # num_hard, 7, 7
    
    # 1. for all pixels: use pmeanError to aggregate
    all_A = torch.pow(torch.pow(predicate_A, p).mean(dim=0), 1.0/p).sum()
    all_B = torch.pow(torch.pow(predicate_B, p).mean(dim=0), 1.0/p).sum()
    # 2. average the clauses
    loss_ex = (all_A + all_B)/(num_class*num_classes + 7*7)
    
    return loss_ex
    
def loss_c(predictions, num_classes, indices_high, eps=1e-8, p=5):
    b, _, h, w = predictions.shape
    predictions = torch.sigmoid(predictions.float())
    
    MCMA = predictions[:,:-7,:,:].permute(0,2,3,1).flatten(0,2) # B*H*W, num_class
    MCMB = predictions[:,-7:,:,:].permute(0,2,3,1).flatten(0,2) # B*H*W, 7
    
    # predicate: 1-p+p*q, with aggregater, simplified to p-p*q
    predicate = MCMA.clone()
    for ii in range(7):
        indices = indices_high[ii]
        predicate[:,indices[0]:indices[1]] = MCMA[:,indices[0]:indices[1]] - MCMA[:,indices[0]:indices[1]]*MCMB[:,ii:ii+1]
        
    # for all clause: use pmeanError to aggregate
    loss_c = torch.pow(torch.pow(predicate, p).mean(dim=0), 1.0/p).sum()/num_classes
    
    return loss_c

def loss_d(predictions, num_classes, indices_high, eps=1e-8, p=5):
    b, _, h, w = predictions.shape
    predictions = torch.sigmoid(predictions.float())
    
    MCMA = predictions[:,:-7,:,:].permute(0,2,3,1).flatten(0,2) # B*H*W, num_class
    MCMB = predictions[:,-7:,:,:].permute(0,2,3,1).flatten(0,2) # B*H*W, 7
    
    # predicate:  1-p+p*q, with aggregater, simplified to p-p*q
    predicate = MCMB.clone()
    for ii in range(7):
        indices = indices_high[ii]
        predicate[:,ii:ii+1] = MCMB[:,ii:ii+1]-MCMB[:,ii:ii+1]*MCMA[:,indices[0]:indices[1]].max(dim=1, keepdim=True)[0]

    # for all clause: use pmeanError to aggregate
    loss_d = torch.pow(torch.pow(predicate, p).mean(dim=0), 1.0/p).sum()/7
    
    return loss_d

@LOSSES.register_module()
class HieraLossCityscapes(nn.Module):

    def __init__(self,
                 num_classes,
                 use_sigmoid=False,
                 loss_weight=1.0,
                 loss_name='hieralosspascalpart'):
        super(HieraLossCityscapes, self).__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.asl = AsymmetricLossOptimized()

    def forward(self,
                cls_score,
                label,
                cls_score_ori,
                step,
                weight=None,
                **kwargs):
        """Forward function."""
        
        targets, targets_top, indices_top = prepare_targets(label)
        
        # build hierarchy GT
        void_indices = (targets==255)
        targets[void_indices]=0
        targets = F.one_hot(targets, self.num_classes).permute(0,3,1,2)
        targets_top = F.one_hot(targets_top, 7).permute(0,3,1,2)
        targets = torch.cat([targets, targets_top], dim=1)
        valid_indices = (~void_indices).unsqueeze(1)
    
        loss = losses_bce_asl(cls_score, targets, valid_indices, self.asl)
#         loss = losses_bce_focal(cls_score, targets, valid_indices)   
        e_rule = loss_e(cls_score_ori, self.num_classes)
        c_rule = loss_c(cls_score_ori, self.num_classes, indices_top)
        d_rule = loss_d(cls_score_ori, self.num_classes, indices_top)
        
        if step<20000:
            factor=0
        elif step<50000:
            factor = float(step-20000)/30000.0
        else:
            factor = 1.0
        
        return loss*self.loss_weight + 0.2*factor*(e_rule+c_rule+d_rule)
    
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

