# model
sam_ckpts = {
    'vit_b': '/x22201018/codes/SAM/checkpoints_sam/sam_vit_b_01ec64.pth',
    'vit_l': '/x22201018/codes/SAM/checkpoints_sam/sam_vit_l_0b3195.pth',
    'vit_h': '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth',
}

sam_type = 'vit_h'
sam_ckpt = sam_ckpts[sam_type]
semantic_module_depth = 2
use_inner_feat = True
inter_idx = 1
use_boundary_head = False


# strategy
max_epochs = 20
loss_type = 'loss_masks'    # loss_masks, alnet
search_lr = [0.05]
warmup_epoch = 5
gamma = 0.9
dice_param = 0.5
save_each_epoch = False

# evaluation
best_score_index = 'pixel'   # 'pixel', 'inst', 'panoptic'
metrics = ['pixel']     # ['pixel', 'inst', 'panoptic']
inst_indicator = 'AJI_plus'      # 'all', 'AJI','AJI_plus','Dice','PQ'
