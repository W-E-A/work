import torch
from mmdet3d.registry import MODELS
from mmengine.model.base_module import BaseModule

@MODELS.register_module()
class Where2commLoss(BaseModule):
    def __init__(self,
                 num_classes: int = 2,
                 cls_weight: float = 1.0,
                 cls_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 reg_weight: float = 2.0,
                 smooth_beta: float = 1.0 / 9.0,
                 use_dir: bool = False):
        super(Where2commLoss, self).__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.cls_alpha = cls_alpha
        self.focal_gamma = focal_gamma
        self.reg_weight = reg_weight
        self.use_dir = use_dir
        self.smooth_beta = smooth_beta
    
    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss
    
    def forward(self, result_dict):
        psm = result_dict['psm'] # [bs, 2, H, W]
        rm = result_dict['rm'] # [bs, 2*7, H, W]
        pos_equal_one = result_dict['pos_equal_one'].to(dtype=psm.dtype) # [bs, H, W, 2]
        labels = result_dict['labels'].to(dtype=rm.dtype) # [bs, H, W, 14]
        batch_size = psm.shape[0]
    
        cls_preds = psm.permute(0, 2, 3, 1).reshape(batch_size, -1, 1).contiguous() # [bs, 2*H*W, 1]
        box_cls_labels = pos_equal_one.reshape(batch_size, -1).contiguous() # [bs, 2*H*W]

        positives = box_cls_labels > 0 # [bs, 2*H*W]
        negatives = box_cls_labels == 0  # [bs, 2*H*W]
        negative_cls_weights = negatives * 1.0 # [bs, 2*H*W]
        cls_weights = (negative_cls_weights + 1.0 * positives).float() # [bs, 2*H*W] all one
        reg_weights = positives.float() # [bs, 2*H*W] pos one
        pos_normalizer = positives.sum(1, keepdim=True).float() # [bs, 1]
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights = cls_weights.unsqueeze(-1)
        reg_weights = reg_weights.unsqueeze(-1)

        one_hot_labels = torch.zeros(
            *list(box_cls_labels.shape), self.num_classes,
            dtype=cls_preds.dtype, device=cls_preds.device
        ) # [bs, 2*H*W, num_classes]
        one_hot_labels.scatter_(-1, box_cls_labels.unsqueeze(dim=-1).long(), 1.0) # [bs, 2*H*W, num_classes] onehot order: neg, pos
        one_hot_labels = one_hot_labels[..., 1:] # [bs, 2*H*W, 1]

        alpha_weight = one_hot_labels * self.cls_alpha + (1 - one_hot_labels) * (1 - self.cls_alpha) # reduce pos loss relatively
        pred_sigmoid = torch.sigmoid(cls_preds) # # [bs, 2*H*W, 1]
        pt = one_hot_labels * (1.0 - pred_sigmoid) + (1.0 - one_hot_labels) * pred_sigmoid # focal bce version
        focal_weight = alpha_weight * torch.pow(pt, self.focal_gamma)

        bce_loss = torch.clamp(pred_sigmoid, min=0) - pred_sigmoid * one_hot_labels + torch.log1p(torch.exp(-torch.abs(pred_sigmoid)))
        cls_loss = focal_weight * bce_loss * cls_weights
        cls_loss = cls_loss.sum() / batch_size * self.cls_weight

        reg_preds = rm.permute(0, 2, 3, 1).reshape(batch_size, -1, 7).contiguous() # [bs, 2*H*W, 7]
        box_reg_labels = labels.reshape(batch_size, -1, 7).contiguous() # [bs, 2*H*W, 7]
        reg_preds_sin, box_reg_labels_sin = self.add_sin_difference(reg_preds, box_reg_labels)
        box_reg_labels_sin = torch.where(torch.isnan(box_reg_labels_sin), reg_preds_sin, box_reg_labels_sin)  # ignore nan targets

        diff = reg_preds_sin - box_reg_labels_sin  # [bs, 2*H*W, 8]

        reg_loss = self.smooth_l1_loss(diff, self.smooth_beta) # [bs, 2*H*W, 8]
        reg_loss *= reg_weights
        reg_loss = reg_loss.sum() / batch_size * self.reg_weight

        # import pdb
        # pdb.set_trace()

        total_loss = cls_loss + reg_loss

        return total_loss, cls_loss, reg_loss