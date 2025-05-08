import torch
from densesam.utils import calc_loss,contrastive_loss,onehot2instmask
from densesam.utils.metrics import find_connect
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmdet.evaluation.functional import INSTANCE_OFFSET
from visual_scripts.pred_draw import draw_building_pred, draw_building_split_pred, draw_cell_pred, draw_boundary_pred, draw_cell_color


def settle_pano(datainfo, pred_inst_mask, device):
    origin_h,origin_w = datainfo.ori_shape
    panoptic_seg = torch.full((origin_h,origin_w), 0, dtype=torch.int32, device=device)
    if np.sum(pred_inst_mask) > 0:
        iid = np.unique(pred_inst_mask)
        for i in iid[1:]:
            mask = pred_inst_mask == i
            panoptic_seg[mask] = (1 + i * INSTANCE_OFFSET)
    datainfo.pred_panoptic_seg = PixelData(sem_seg=panoptic_seg[None])

def train_one_epoch(model, train_loader, optimizer, logger, device, cfg, print_interval):

    model.train()
    len_loader = len(train_loader)
    
    for i_batch, sampled_batch in enumerate(tqdm(train_loader, ncols=70)):

        gt_foreground_masks = torch.stack(sampled_batch['gt_foreground_masks']).to(device).float()
        outputs = model(sampled_batch)
        pred_logits = outputs['logits_256']  # (bs, k, o_h, o_w)
        bs,h,w = gt_foreground_masks.shape
        pred_logits = F.interpolate(
            pred_logits,
            (h,w),
            mode="bilinear",
            align_corners=False,
        )

        if cfg.use_boundary_head:
            gt_boundary_masks = torch.stack(sampled_batch['gt_boundary_mask']).to(device).float()    # (bs, h, w)
            # targets = torch.stack((gt_foreground_masks, gt_boundary_masks),dim=1).reshape(-1, h, w)
            targets = (gt_foreground_masks, gt_boundary_masks)
        else:
            targets = (gt_foreground_masks, None)
        
        loss = calc_loss(pred_logits=pred_logits, target_masks=targets, args=cfg)

        contrast_feat = outputs['contrast_feat']    # (bs, h*w, c)
        if contrast_feat is not None:
            token_gt = F.interpolate(gt_foreground_masks.unsqueeze(1),(64,64),mode="nearest")
            token_gt = token_gt.flatten(1)    # (bs, h*w)
            cont_loss = contrastive_loss(contrast_feat, token_gt, cfg.temperature)
            loss = loss + 0.1*cont_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i_batch+1) % print_interval == 0:
            info_str = f'iteration {i_batch+1}/{len_loader}, loss: {loss.item():.6f}'
            logger.info(info_str)

def val_one_epoch(model, val_loader, evaluators, device, cfg, visual_args=None):
    
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_loader, ncols=70)):
        with torch.no_grad():
            outputs = model(sampled_batch)
        logits = outputs['logits_256']  # (bs or k_prompt, 3, 256, 256)

        # specified_image_name = 'image_11_2.png'
        # all_names = [item.img_path.split('/')[-1] for item in sampled_batch['data_samples']]
        # if specified_image_name not in all_names:
        #     continue

        all_datainfo = []
        for idx,datainfo in enumerate(sampled_batch['data_samples']):
            origin_size = datainfo.ori_shape
            logits_origin = F.interpolate(
                logits[idx].unsqueeze(0),
                origin_size,
                # (1024,1024),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)    # (k,o_h,o_w)
            
            pred_foreground_mask = (logits_origin[0].detach() > 0).type(torch.uint8) # (o_h,o_w)
            
            need_inst_with_noboundary = not cfg.use_boundary_head and ('panoptic' in cfg.metrics or 'inst' in cfg.metrics)
            if need_inst_with_noboundary:
                pred_inst_mask = find_connect(pred_foreground_mask, dilation=False)
            if cfg.use_boundary_head:
                pred_result = (
                    torch.sigmoid(logits_origin[0,:,:].detach().unsqueeze(0)), 
                    torch.sigmoid(logits_origin[1,:,:].detach().unsqueeze(0))
                )
                pred_inst_mask, pred_boundary_mask = model.postprocess(datainfo, pred_result)
                pred_foreground_mask = pred_inst_mask > 0

            if 'panoptic' in cfg.metrics:
                settle_pano(datainfo, pred_inst_mask, device)
                    
            datainfo.pred_sem_seg = PixelData(sem_seg=torch.as_tensor(pred_foreground_mask[None]))
            all_datainfo.append(datainfo.to_dict())

            if 'inst' in cfg.metrics:
                gt_inst_seg = datainfo.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
                gt_inst_mask = onehot2instmask(gt_inst_seg)
                evaluators['inst_evaluator'].process(gt_inst_mask, pred_inst_mask)
            
            if visual_args is not None and visual_args['visual_pred'] and i_batch % visual_args['visual_interval']==0:
                pred_info = dict(pred_mask = pred_foreground_mask)
                pred_save_dir = visual_args['pred_save_dir']
                draw_func = visual_args['draw_func']
                if draw_func == 'draw_building_pred':
                    draw_building_pred(datainfo, pred_info, pred_save_dir)
                if draw_func == 'draw_building_split_pred':
                    draw_building_split_pred(datainfo, pred_info, pred_save_dir)
                
                if draw_func == 'draw_cell_pred':
                    gt_info = dict(gt_inst_mask = gt_inst_mask)
                    pred_info['pred_inst_mask'] = pred_inst_mask
                    draw_cell_pred(datainfo, gt_info, pred_info, pred_save_dir)
                if draw_func == 'draw_cell_color':
                    gt_info = dict(gt_inst_mask = gt_inst_mask)
                    pred_info['pred_inst_mask'] = pred_inst_mask
                    draw_cell_color(datainfo, gt_info, pred_info, pred_save_dir)
                if draw_func == 'draw_boundary_pred':
                    gt_info = dict(gt_inst_mask = gt_inst_mask, 
                                   gt_boundary = sampled_batch['gt_boundary_mask'][idx])
                    pred_info['pred_inst_mask'] = pred_inst_mask
                    pred_info['pred_boundary'] = pred_boundary_mask
                    draw_boundary_pred(datainfo, gt_info, pred_info, pred_save_dir)
        
        if 'pixel' in cfg.metrics:
            evaluators['semantic_evaluator'].process(None, all_datainfo)
        if 'panoptic' in cfg.metrics:
            evaluators['panoptic_evaluator'].process(None, all_datainfo)
            
    total_datas = len(val_loader)*cfg.val_bs
    metrics = dict()
    if 'pixel' in cfg.metrics:
        semantic_metrics = evaluators['semantic_evaluator'].evaluate(total_datas)
        metrics['semantic'] = semantic_metrics
    if 'panoptic' in cfg.metrics:
        panoptic_metrics = evaluators['panoptic_evaluator'].evaluate(total_datas)
        metrics['panoptic'] = panoptic_metrics
    if 'inst' in cfg.metrics:
        inst_metrics = evaluators['inst_evaluator'].evaluate(total_datas)
        metrics['inst'] = inst_metrics

    return metrics
