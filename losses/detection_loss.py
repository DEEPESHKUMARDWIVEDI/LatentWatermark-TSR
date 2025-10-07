import torch
import torch.nn as nn

class DetectionLoss(nn.Module):
    """
    Cross entropy loss for watermark region segmentation
    """
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred_mask, gt_mask):
        """
        pred_mask: [B, C, H, W] logits
        gt_mask: [B, H, W] long tensor with class indices
        """
        return self.criterion(pred_mask, gt_mask)
