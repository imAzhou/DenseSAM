'''
    paper: AL-Net: Attention Learning Network Based on Multi-Task Learning for Cervical Nucleus Segmentation
    code: https://github.com/jingzhaohlj/AL-Net/blob/main/loss.py
'''

import torch.nn as nn
import torch
import torch.nn.functional as F

def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1
    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )
    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )
    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)
    return unfolded_x

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def consistency(self,y_true,  y_pred):
        a = self.bce_loss(y_pred, y_true)
        sloss = y_pred.clone().detach()
        thred = y_pred.clone().detach()
        unfolded_images = unfold_wo_center(
            sloss, kernel_size=3, dilation=2
        )            
        diff = abs(sloss[:, :, None] - unfolded_images)
        unfolded_weights = torch.max(diff, dim=2)[0]
        thred[thred<0.5] = 0
        loss1 = unfolded_weights * thred
        thred[thred>=0.5] = 1
        loss=loss1 + a
        return loss.mean()

    def __call__(self, y_true, y_pred):
        '''
            y_true: tensor, shape is (bs,1,h,w), torch.float32
            y_pred: tensor, shape is (bs,1,h,w), torch.float32
        '''
        b = self.consistency(y_true, y_pred)
        return b
