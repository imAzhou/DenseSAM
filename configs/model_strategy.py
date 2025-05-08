# model
sam_ckpts = {
    'vit_b': 'checkpoints_sam/sam_vit_b_01ec64.pth',
    'vit_l': 'checkpoints_sam/sam_vit_l_0b3195.pth',
    'vit_h': 'checkpoints_sam/sam_vit_h_4b8939.pth',
}

sam_type = 'vit_h'
sam_ckpt = sam_ckpts[sam_type]
use_local = True
use_global = True
use_inner_idx = [1]
use_boundary_head = True
temperature = 0.1  # 较低值会导致对比损失中的相似度差异更为明显，从而加速模型的收敛，但也可能导致梯度爆炸

# strategy
max_epochs = 30
loss_type = 'loss_masks'    # loss_masks, alnet
search_lr = [0.001]
warmup_epoch = 5
gamma = 0.95
dice_param = 0.5
save_each_epoch = False

# evaluation
best_score_index = 'inst'   # 'pixel', 'inst', 'panoptic'
metrics = ['inst']     # ['pixel', 'inst', 'panoptic']
inst_indicator = 'AJI_plus'      # 'all', 'AJI','AJI_plus','Dice','PQ'
