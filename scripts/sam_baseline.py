import os
import torch
import argparse
from utils import get_prompt,show_points,show_box
from densesam.datasets.create_loader import gene_loader_trainval
from mmengine.config import Config
from utils.metrics import get_metrics
from mmdet.evaluation.functional import INSTANCE_OFFSET
from tqdm import tqdm
import numpy as np
import cv2
from densesam.models.dense_sam import SAMBaselineNet
import matplotlib.pyplot as plt
import copy
import random

parser = argparse.ArgumentParser()

# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--vis_result_save_dir', type=str)
parser.add_argument('--prompt_type', type=str)
parser.add_argument('--seed', type=int, default=666, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--visual_pred', action='store_true')

args = parser.parse_args()

semantic_sets = ['whu', 'inria', 'mass']
instance_sets = ['conic', 'monuseg', 'cpm17']

def onehot2instmask(onehot_mask):
    h,w = onehot_mask.shape[1],onehot_mask.shape[2]
    instmask = torch.zeros((h,w), dtype=torch.int32)
    for iid,seg in enumerate(onehot_mask):
        instmask[seg.astype(bool)] = iid+1
    
    return instmask

def remap_mask(inst_mask):
    remap_inst_mask = np.zeros_like(inst_mask)
    iid = np.unique(inst_mask)
    for i,idx in enumerate(iid[1:]):
        remap_inst_mask[inst_mask == idx] = i+1
    return remap_inst_mask

def find_connect(image_mask_np):
    '''
    Args:
        image_mask_np(numpy): The image masks. Expects an
            image in HW uint8 format, with pixel values in [0, 1].
    return:
        [[x1,y1,x2,y2], ..., [x1,y1,x2,y2]]
    '''
    # 寻找连通域
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mask_np, connectivity=8)
    trans_box = lambda x1,y1,w,h: [x1,y1, x1 + w, y1 + h]
    # stats[0] 是背景框
    all_boxes = np.array([trans_box(x1,y1,w,h) for x1,y1,w,h,_ in stats[1:]])
    return torch.as_tensor(all_boxes), torch.as_tensor(labels), torch.as_tensor(centroids)

def draw_visual_result(draw_items, is_semantic, datainfo, ratio):

    img_path = datainfo.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h,w = draw_items['pred_inst_mask'].shape
    
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(221)
    ax.imshow(img)
    # ax.axhline(y=h//2, color='black')
    # 或者绘制一条在x=30位置的垂直黑色直线
    # ax.axvline(x=w//2, color='black')

    ax.set_axis_off()
    ax.set_title('input image')
    if draw_items['point_prompt'] is not None:
        coords_torch,labels_torch = draw_items['point_prompt']
        coords_torch /= ratio
        show_points(coords_torch.cpu(), labels_torch.cpu(), ax, marker_size=400)
    if draw_items['box_prompt'] is not None:
        boxes = draw_items['box_prompt'].squeeze(1).cpu()
        boxes /= ratio
        for box in boxes:
            show_box(box, ax)
    
    ax = fig.add_subplot(222)
    
    show_color_gt = np.zeros((h,w,4))
    show_color_gt[:,:,3] = 1
    inst_nums = len(np.unique(draw_items['gt_inst_mask'])) - 1
    for i in range(inst_nums):
        color_mask = np.concatenate([np.random.random(3), [1]])
        show_color_gt[draw_items['gt_inst_mask']==i+1] = color_mask
    ax.imshow(show_color_gt)
    # ax.axhline(y=h//2, color='white')
    # # 或者绘制一条在x=30位置的垂直黑色直线
    # ax.axvline(x=w//2, color='white')
    ax.set_title('gt inst mask')

    ax = fig.add_subplot(223)
    ax.set_axis_off()
    if is_semantic:
        ax.imshow(draw_items['pred_sema_mask'], cmap='gray')
    else:
        h,w = draw_items['pred_inst_mask'].shape
        show_color_pred = np.zeros((h,w,4))
        show_color_pred[:,:,3] = 1
        inst_nums = len(np.unique(draw_items['pred_inst_mask'])) - 1
        for i in range(inst_nums):
            color_mask = np.concatenate([np.random.random(3), [1]])
            show_color_pred[draw_items['pred_inst_mask']==i+1] = color_mask
        ax.imshow(show_color_pred)
        # ax.axhline(y=h//2, color='white')
        # # 或者绘制一条在x=30位置的垂直黑色直线
        # ax.axvline(x=w//2, color='white')
        ax.set_title('pred inst mask')
    
    ax = fig.add_subplot(224)
    ax.imshow(draw_items['embed_256'], cmap='hot')
    ax.set_title('upsample embed')
    
    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()

def draw_patch_inner_embed(draw_items, datainfo, ratio):
    img_path = datainfo.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h,w,_ = image.shape
    
    vis_embeds_64 = draw_items['vis_embeds_64']
    embeds_64_var = draw_items['embeds_64_var']
    columns = len(vis_embeds_64) + 1
    
    fig = plt.figure(figsize=(3*columns,8))
    ax = fig.add_subplot(2,columns,1)
    ax.imshow(img)
    grid_size = 32
    ax.set_xticks(np.arange(0, h, grid_size))
    ax.set_yticks(np.arange(0, w, grid_size))
    ax.grid(color='black', linestyle='-', linewidth=1)
    # ax.set_axis_off()

    ax = fig.add_subplot(2,columns,columns+1)
    # ax.imshow(img)

    gt_inst_mask = draw_items['gt_inst_mask']
    inst_nums = len(np.unique(gt_inst_mask)) - 1
    copy_img = copy.deepcopy(img)
    for i in range(inst_nums):
        color_mask = [random.randint(0, 255) for _ in range(3)]
        copy_img[gt_inst_mask==i+1] = color_mask
    ax.imshow(copy_img)

    if draw_items['point_prompt'] is not None:
        coords_torch,labels_torch = draw_items['point_prompt']
        coords_torch /= ratio
        show_points(coords_torch.cpu(), labels_torch.cpu(), ax, marker_size=400)
    if draw_items['box_prompt'] is not None:
        boxes = draw_items['box_prompt'].squeeze(1).cpu()
        boxes /= ratio
        for box in boxes:
            show_box(box, ax)

    for i in range(len(vis_embeds_64)):
        ax = fig.add_subplot(2,columns,i+2)

        if draw_items['point_prompt'] is not None:
            coords_torch,labels_torch = draw_items['point_prompt']
            coords_torch /= ratio
            show_points(coords_torch.cpu(), labels_torch.cpu(), ax, marker_size=400)
        if draw_items['box_prompt'] is not None:
            boxes = draw_items['box_prompt'].squeeze(1).cpu()
            boxes /= ratio
            for box in boxes:
                show_box(box//2, ax)
        
        ax.imshow(vis_embeds_64[i], cmap='hot')
        ax.set_title(f'inner embed {i} mean')
        # ax.set_axis_off()
    
    for i in range(len(embeds_64_var)):
        ax = fig.add_subplot(2,columns,i+2+columns)
        ax.imshow(embeds_64_var[i], cmap='hot')
        ax.set_title(f'inner embed {i} var')
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()

def eval_sam():
    model.eval()
    for i_batch, sampled_batch in enumerate(tqdm(val_dataloader, ncols=70)):

        # [473, 143],[457,21],[341,204],[403,126], [1,1,1,0]
        # specified_points = np.array([[473, 143]]) # [x1,y1]
        # specified_points_label = np.array([1])
        # specified_points = np.array([[72,192], [404,138], [275,381], [196,263]]) # [x1,y1]
        specified_points = np.array([[206,193]]) # [x1,y1]
        specified_points_label = np.array([1])
        specified_bbox = np.array([165,149, 241,243])
        # specified_image_name = 'image_03_0.png'
        # specified_image_name = 'image_05_2.png'
        specified_image_name = 'image_00_3.png'

        data_sample = sampled_batch['data_samples'][0]
        img_path = data_sample.img_path
        image_name = img_path.split('/')[-1]
        if image_name != specified_image_name:
            continue
            
        gt_sem_seg = data_sample.gt_sem_seg.sem_seg[0].to(torch.int8).numpy()
        origin_h,origin_w = sampled_batch['data_samples'][0].ori_shape
        if cfg.dataset_tag in semantic_sets:
            gt_bboxes,gt_inst_mask,gt_centroids = find_connect(gt_sem_seg)
        else:
            gt_bboxes = data_sample.gt_instances.bboxes.tensor  # tensor, (inst_nums, 4)
            gt_centroids = data_sample.gt_instances.bboxes.centers
            gt_inst_seg = data_sample.gt_instances.masks.masks  # np.array, (inst_nums, h, w)
            gt_inst_mask = onehot2instmask(gt_inst_seg)  # tensor, (h, w)   
        
        panoptic_seg = torch.full((origin_h,origin_w), 0, dtype=torch.int32, device=device)
        pred_sema_mask = torch.full((origin_h,origin_w), 0, dtype=torch.int8, device=device)
        pred_inst_mask = np.zeros((origin_h,origin_w), dtype=int)
        if len(gt_bboxes) > 0:
            scale = 1024 // origin_h
            if args.prompt_type == 'specified_points':
                box_prompt = None
                specified_points = specified_points * scale
                coords_torch = torch.as_tensor(specified_points, dtype=torch.float, device=device)
                labels_torch = torch.as_tensor(specified_points_label, dtype=torch.int, device=device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                point_prompt = (coords_torch, labels_torch)
            elif args.prompt_type == 'specified_bbox':
                point_prompt = None
                box_torch = torch.as_tensor(specified_bbox[None, :] * scale, dtype=torch.float, device=device)  #[[x1,y1,x2,y2]]
                box_prompt = box_torch[None, :]  #[[[x1,y1,x2,y2]]]
            elif args.prompt_type == 'empty':
                box_prompt,point_prompt = None, None
            else:
                box_prompt,point_prompt,_ = get_prompt(args.prompt_type, gt_sem_seg, gt_bboxes, device, scale)
            prompt_dict = dict(point_prompt=point_prompt,box_prompt=box_prompt)
            outputs = model(sampled_batch, prompt_dict)
            
            # pred_mask.shape: (box_nums, h, w)
            pred_mask = (outputs['logits_origin'].detach() > 0).squeeze(1)
            for iid, inst_mask in enumerate(pred_mask):
                iid += 1
                panoptic_seg[inst_mask] = (1 + iid * INSTANCE_OFFSET)
                pred_inst_mask[inst_mask.cpu().numpy()] = iid
                pred_sema_mask[inst_mask] = 1
            
            if args.visual_pred:
                sam_img_embed_256 = outputs['embeddings_256'][0]  # (32, 256, 256)
                embed_256 = torch.mean(sam_img_embed_256, dim=0).detach().cpu().numpy()
                embed_256_var = torch.var(sam_img_embed_256, dim=0).detach().cpu().numpy()
                # draw_items = dict(
                #     pred_sema_mask = pred_sema_mask.cpu(), pred_inst_mask = pred_inst_mask,
                #     gt_inst_mask = gt_inst_mask, embed_256 = embed_256,
                #     point_prompt = point_prompt, box_prompt = box_prompt
                # )
                # draw_visual_result(draw_items, cfg.dataset_tag in semantic_sets, data_sample,scale)
                
                # sam_inter_features = outputs['inter_features']  # [(bs=1,256, 256, 1280),...]
                # sam_inter_features = outputs['decoder_inner_embeds']  # [(bs=1,256, 256, 256),...]
                # vis_embeds_64 = []
                # embeds_64_var = []
                # for sam_embed_64 in sam_inter_features:
                #     vis_sam_embed_64 = torch.mean(sam_embed_64[0], dim=2).detach().cpu().numpy()
                #     sam_embed_64_var = torch.var(sam_embed_64[0], dim=2).detach().cpu().numpy()
                #     vis_embeds_64.append(vis_sam_embed_64)
                #     embeds_64_var.append(sam_embed_64_var)
                draw_items = dict(
                    vis_embeds_64 = [embed_256],
                    # vis_embeds_64 = vis_embeds_64,
                    embeds_64_var = [embed_256_var],
                    # embeds_64_var = embeds_64_var,
                    point_prompt = point_prompt, box_prompt = box_prompt,
                    gt_inst_mask = gt_inst_mask
                )
                draw_patch_inner_embed(draw_items, data_sample, scale)
            
        # data_sample.pred_panoptic_seg = PixelData(sem_seg = panoptic_seg[None])
        # data_sample.pred_sem_seg = PixelData(sem_seg = pred_sema_mask[None])

        # evaluators['semantic_evaluator'].process(None, [data_sample.to_dict()])
        # evaluators['panoptic_evaluator'].process(None, [data_sample.to_dict()])
        # pred_inst_mask = remap_mask(pred_inst_mask)
        # evaluators['inst_evaluator'].process(gt_inst_mask, pred_inst_mask)
    
    # total_datas = len(val_dataloader)*cfg.val_bs
    # semantic_metrics = evaluators['semantic_evaluator'].evaluate(total_datas)
    # panoptic_metrics = evaluators['panoptic_evaluator'].evaluate(total_datas)
    # inst_metrics = evaluators['inst_evaluator'].evaluate(total_datas)
    # metrics = dict(
    #     semantic = semantic_metrics,
    #     panoptic = panoptic_metrics,
    #     inst = inst_metrics,
    # )
    return None


def main(logger_name, cfg):

        # val
        metrics = eval_sam()
        # print(metrics)



if __name__ == "__main__":
    device = torch.device(args.device)

    d_cfg = Config.fromfile(args.dataset_config_file)
    model_strategy_config_file = 'configs/model_strategy.py'
    m_s_cfg = Config.fromfile(model_strategy_config_file)

    cfg = Config()
    for sub_cfg in [d_cfg, m_s_cfg]:
        cfg.merge_from_dict(sub_cfg.to_dict())
    
     # register model
    model = SAMBaselineNet(
        use_embed = cfg.dataset.load_embed,
        sam_ckpt = cfg.sam_ckpt,
        sam_type = cfg.sam_type,
        device = device
    ).to(device)

    # load datasets
    train_dataloader, val_dataloader, metainfo, restinfo = gene_loader_trainval(
        dataset_config = d_cfg, seed = args.seed)

    # get evaluator
    evaluators = get_metrics(cfg.metrics, metainfo, restinfo, cfg.inst_indicator)

    if args.visual_pred:
        pred_save_dir = args.vis_result_save_dir
        os.makedirs(pred_save_dir, exist_ok = True)
        eval_sam()


'''
specified_points specified_bbox random_bbox all_bboxes empty

python scripts/sam_baseline.py \
    configs/datasets/cpm17.py \
    --record_save_dir logs/debug \
    --prompt_type specified_points \
    --vis_result_save_dir visual_results/sam_baseline_output/cpm17_image_00_3 \
    --visual_pred
'''