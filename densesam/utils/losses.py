import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from mmdet.models.losses import FocalLoss,DiceLoss
from .loss_mask import loss_masks
from .consistency_loss import ConsistencyLoss


def calc_loss(*, pred_logits, target_masks, args):
    '''
    Args:
        pred_logits: (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for class-specific predict logits.
        target_masks: A tensor of shape (N, H, W) that contains class index on a H x W grid, 0 mean bg, valued class id from 1 to 2,3,4....,C-1
    '''

    if args.loss_type == 'loss_masks':  # sample points calc loss, only for binary mask
        bs,c,h,w = pred_logits.shape
        pred_logits = pred_logits.reshape(-1, h, w).unsqueeze(1)
        gt_foreground_masks, gt_boundary_masks = target_masks
        if gt_boundary_masks is not None:
            targets = torch.stack((gt_foreground_masks, gt_boundary_masks),dim=1).reshape(-1, h, w)
        else:
            targets = gt_foreground_masks
        bce_loss, dice_loss = loss_masks(pred_logits, targets.unsqueeze(1))
        loss = (1 - args.dice_param) * bce_loss + args.dice_param * dice_loss
        return loss
    if args.loss_type == 'focal_dice':  # only for binary mask
        return focal_dice(pred_logits, target_masks.unsqueeze(1), args)
    if args.loss_type == 'ce_dice': # calc every pixels loss
        return ce_dice(pred_logits, target_masks, args)
    
    if args.loss_type == 'alnet':
        pred_mask_logits = torch.sigmoid(pred_logits[:,0,...].unsqueeze(1))
        pred_boundary_logits = torch.sigmoid(pred_logits[:,1,...].unsqueeze(1))
        gt_foreground_masks, gt_boundary_masks = target_masks

        mask_loss = ConsistencyLoss()(gt_foreground_masks.unsqueeze(1), pred_mask_logits)
        boundary_loss = nn.BCELoss()(pred_boundary_logits, gt_boundary_masks.unsqueeze(1))
        loss = mask_loss + boundary_loss
        return loss

def focal_dice(pred_logits, target_masks, args):
    focal_loss_fn = FocalLoss()
    bdice_loss_fn = DiceLoss()
    bce_loss = focal_loss_fn(pred_logits, target_masks)
    dice_loss = bdice_loss_fn(pred_logits.flatten(1), target_masks.flatten(1))
    loss = (1 - args.dice_param) * bce_loss + args.dice_param * dice_loss
    return loss

def ce_dice(pred_logits, target_masks, args):
    '''
    for CrossEntropyLoss, target_masks each pixel record its class id,
    for DiceLoss, target_masks shoud be converted to onehot format.
    '''
    ignore_index = 255
    ce_loss_fn = CrossEntropyLoss(ignore_index = ignore_index)
    ce_loss = ce_loss_fn(pred_logits, target_masks.long())

    dice_loss_fn = DiceLoss()
    cls_nums = pred_logits.shape[1]
    gt_c_mask = torch.zeros_like(pred_logits).to(pred_logits.device)
    for cid in range(cls_nums):
        gt_c_mask[:,cid,:,:] = (target_masks == cid).int()
    dice_loss = dice_loss_fn(pred_logits,gt_c_mask)
    if torch.sum(target_masks != 255) != 0:
        loss = (1 - args.dice_param) * ce_loss + args.dice_param * dice_loss
    else:
        loss = torch.tensor(0.0, requires_grad=True)
    return loss

def contrastive_loss(features, labels, temperature=0.07):
    """
    计算监督对比学习损失。
    
    Args:
        features: Tensor of shape (bs, num_tokens, embed_dim).
        labels: Tensor of shape (bs, num_tokens) with integer class labels.
        temperature: Temperature scaling factor for contrastive loss.
    
    Returns:
        loss: Supervised contrastive loss (scalar).
    """
    bs, num_tokens, embed_dim = features.shape
    
    # Step 2: Reshape to combine batch and num_tokens for easier pairwise similarity computation
    features = features.view(bs * num_tokens, embed_dim)  # (bs * num_tokens, embed_dim)
    labels = labels.view(-1)  # (bs * num_tokens)
    
    # Step 3: Compute pairwise similarity (dot product)
    similarity_matrix = torch.matmul(features, features.T)  # (bs * num_tokens, bs * num_tokens)
    exp_sim = torch.exp(similarity_matrix / temperature)
    
    # Step 4: Create a mask to identify positive pairs
    label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # (bs * num_tokens, bs * num_tokens)
    mask_self = torch.eye(label_mask.size(0), device=features.device)
    positive_mask = label_mask - mask_self  # Remove diagonal (self-pairs)

    # Step 5: Compute log probabilities for positive pairs
    log_prob = similarity_matrix / temperature - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Step 6: Sum log probabilities for positive samples
    positive_log_prob = positive_mask * log_prob
    positive_count = positive_mask.sum(dim=1)
    
    loss = -(positive_log_prob.sum(dim=1) / positive_count.clamp(min=1))  # Avoid division by zero
    return loss.mean()
