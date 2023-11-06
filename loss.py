from symbol import or_test
from cv2 import threshold
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HourGlassLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(HourGlassLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predict, target):
        predict = torch.clamp(predict, min=1e-4, max=1-1e-4)

        pos_inds = target.gt(0.9)
        neg_inds = target.lt(0.9)
        neg_weights = torch.pow(1 - target[neg_inds], 4)

        pos_pred = predict[pos_inds]
        neg_pred = predict[neg_inds]

        pos_loss = torch.log2(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log2(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        

        return loss


