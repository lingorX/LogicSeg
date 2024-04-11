import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

TORCH_VERSION = torch.__version__[:3]

_euler_num = 2.718281828  # euler number
_pi = 3.14159265  # pi
_ln_2_pi = 1.837877  # ln(2 * pi)
_CLIP_MIN = 1e-6  # min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0  # max clip value after softmax or sigmoid operations
_POS_ALPHA = 1e-3  # add this factor to ensure the AA^T is positive definite
_IS_SUM = 1  # sum the loss per channel


def map_get_pairs(labels_4D, probs_4D, radius=3, is_combine=True):

    # the original height and width
    label_shape = labels_4D.size()
    h, w = label_shape[2], label_shape[3]
    new_h, new_w = h - (radius - 1), w - (radius - 1)
    
    # get the neighbors
    la_ns = []
    pr_ns = []
    # for x in range(0, radius, 1):
    for y in range(0, radius, 1):
        for x in range(0, radius, 1):
            la_now = labels_4D[:, :, y:y + new_h, x:x + new_w]
            pr_now = probs_4D[:, :, y:y + new_h, x:x + new_w]
            la_ns.append(la_now)
            pr_ns.append(pr_now)

    if is_combine:
        # for calculating RMI
        pair_ns = la_ns + pr_ns
        p_vectors = torch.stack(pair_ns, dim=2)
        return p_vectors
    else:
        # for other purpose
        la_vectors = torch.stack(la_ns, dim=2)
        pr_vectors = torch.stack(pr_ns, dim=2)
        return la_vectors, pr_vectors


def log_det_by_cholesky(matrix):
    # This uses the property that the log det(A) = 2 * sum(log(real(diag(C))))
    # where C is the cholesky decomposition of A.
    chol = torch.cholesky(matrix)
    # return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-6), dim=-1)
    return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)


class RMILoss(nn.Module):
    """
    region mutual information
    I(A, B) = H(A) + H(B) - H(A, B)
    This version need a lot of memory if do not dwonsample.
    """

    def __init__(self,
                 num_classes=26,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3):
        super(RMILoss, self).__init__()
        self.num_classes = num_classes
        # radius choices
        self.rmi_radius = rmi_radius
        assert self.rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_pool_way = rmi_pool_way
        assert self.rmi_pool_way in [0, 1, 2, 3]

        # set the pool_size = rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        assert self.rmi_pool_size == self.rmi_pool_stride
        
        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = 255

    def forward(self,
                cls_score,
                new_targets,
                valid_indices):
        loss = self.forward_sigmoid(cls_score, new_targets, valid_indices)
        return loss
    
    def forward_sigmoid(self, logits_4D, labels_4D, label_mask_3D):
        # valid label
        valid_onehot_labels_4D = labels_4D.float()
        label_mask_3D = label_mask_3D.float()
        label_mask_flat = label_mask_3D.view([-1, ])
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=1)
        valid_onehot_labels_4D.requires_grad_(False)
        
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + _CLIP_MIN
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)
        return rmi_loss

    def rmi_lower_bound(self, labels_4D, probs_4D):
        assert labels_4D.size() == probs_4D.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.max_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(new_h, new_w), mode='bilinear', align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]
        la_vectors, pr_vectors = map_get_pairs(labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)
        la_vectors = la_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor).requires_grad_(False)
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor)
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        appro_var = la_cov - torch.matmul(la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1))
        rmi_now = 0.5 * log_det_by_cholesky(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))
        rmi_loss = torch.sum(rmi_per_class) if _IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss