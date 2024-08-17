import torch
import os
import argparse
from utils import set_seed,show_points,show_box,show_mask
from models.dense_sam import DenseSAMNet
import matplotlib.pyplot as plt
import cv2
import numpy as np
from panopticapi.utils import rgb2id
import random
import torch.nn.functional as F

parser = argparse.ArgumentParser()

# base args
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

def get_color_inst_mask(inst_mask):
    h,w = inst_mask.shape
    show_color_mask = np.zeros((h,w,4))
    show_color_mask[:,:,3] = 1    # opacity
    inst_nums = len(np.unique(inst_mask)) - 1
    for i in range(inst_nums):
        color_mask = np.concatenate([np.random.random(3), [1]])
        show_color_mask[inst_mask==i+1] = color_mask
    
    return show_color_mask


def gtmap2inst(gt_mask):
    h,w,_ = gt_mask.shape
    result = np.zeros((h,w))
    id_mask = rgb2id(gt_mask)
    zero_mask = (gt_mask == 255)[:,:,0]
    bg_id = id_mask[zero_mask][0]
    iid = np.unique(id_mask)
    for i, id in enumerate(iid):
        if id != bg_id:
            result[id_mask == id] = i+1
    
    return result

def sample_prompt(inst_mask):
    iid = np.unique(inst_mask)
    random_ids = random.sample(list(iid[1:]), 3)
    points,bboxes,masks = [],[],[]
    for random_id in random_ids:
        mask = (inst_mask == random_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        x1, y1, w, h = cv2.boundingRect(contour)
        bbox = np.array([x1, y1, x1+w, y1+h])
        point = np.array([int(x1 + w/2), int(y1 + h/2)])
        points.append(point)
        bboxes.append(bbox)
        masks.append(mask)

    return points,bboxes,masks


def main():
    sam_ckpt = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth'
    save_dir = f'visual_results/sam_embed_in_prompt'
    root_dir = '/x22201018/datasets/MedicalDatasets/CPM17/test_p512'
    image_name = 'image_03_0'
    vis_png = f'{root_dir}/img_dir/{image_name}.png'
    gt_mask = f'{root_dir}/panoptic_seg_anns_coco/{image_name}.png'
    vis_embed = f'{root_dir}/img_tensor/{image_name}.pt'

    img_mask = cv2.imread(gt_mask)
    h,w = img_mask.shape[:2]
    scale = 1024//h
    gt_inst_mask = gtmap2inst(img_mask)
    points,bboxes,masks = sample_prompt(gt_inst_mask)

    point = points[0]
    bbox = bboxes[1]    # [x1,y1,x2,y2]
    mask_bbox = bboxes[2]

    point_prompt = (
        torch.as_tensor(np.array([point*scale]), dtype=torch.float, device=device).unsqueeze(0),
        torch.ones(1, dtype=torch.int, device=device).unsqueeze(0)
    )
    bbox_prompt = torch.as_tensor(bbox[None, :]*scale, dtype=torch.float, device=device).unsqueeze(0)
    mask_bbox_prompt = torch.as_tensor(mask_bbox[None, :]*scale, dtype=torch.float, device=device).unsqueeze(0)
    # mask_prompt = F.interpolate(
    #     torch.as_tensor(mask*20, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0),
    #     (256,256)
    # )

    # register model
    sam_model = DenseSAMNet(
                num_classes = -1,
                num_mask_tokens = 4,
                sm_depth = 0,
                use_inner_feat = False,
                use_embed = True,
                sam_ckpt = sam_ckpt,
                device = device
            ).to(device)

    bs_image_embedding = torch.load(vis_embed).to(device).unsqueeze(0)
    image_pe = sam_model.prompt_encoder.get_dense_pe()
    sparse_point,dense_point = sam_model.prompt_encoder(
        points = point_prompt,
        boxes = None,
        masks = None,
    )
    point_outputs = sam_model.mask_decoder(
        image_embeddings = bs_image_embedding,
        image_pe = image_pe,
        sparse_prompt_embeddings = sparse_point,
        dense_prompt_embeddings = dense_point,
        inter_feature = None,
        multimask_output = True
    )
    sparse_bbox,dense_bbox = sam_model.prompt_encoder(
        points = None,
        boxes = bbox_prompt,
        masks = None,
    )
    bbox_outputs = sam_model.mask_decoder(
        image_embeddings = bs_image_embedding,
        image_pe = image_pe,
        sparse_prompt_embeddings = sparse_bbox,
        dense_prompt_embeddings = dense_bbox,
        inter_feature = None,
        multimask_output = True
    )
    sparse_mask_bbox,dense_mask_bbox = sam_model.prompt_encoder(
        points = None,
        boxes = mask_bbox_prompt,
        masks = None,
    )
    mask_bbox_outputs = sam_model.mask_decoder(
        image_embeddings = bs_image_embedding,
        image_pe = image_pe,
        sparse_prompt_embeddings = sparse_mask_bbox,
        dense_prompt_embeddings = dense_mask_bbox,
        inter_feature = None,
        multimask_output = True
    )
    bbox_logits = mask_bbox_outputs['logits_256']  # (bs, 4, 256, 256)
    mask_prompt = bbox_logits[:,0,:,:].unsqueeze(1).detach()
    sparse_mask,dense_mask = sam_model.prompt_encoder(
        points = None,
        boxes = None,
        masks = mask_prompt,
    )
    mask_outputs = sam_model.mask_decoder(
        image_embeddings = bs_image_embedding,
        image_pe = image_pe,
        sparse_prompt_embeddings = sparse_mask,
        dense_prompt_embeddings = dense_mask,
        inter_feature = None,
        multimask_output = True
    )
    sparse_noprompt,dense_noprompt = sam_model.prompt_encoder(
        points = None,
        boxes = None,
        masks = None,
    )
    noprompt_outputs = sam_model.mask_decoder(
        image_embeddings = bs_image_embedding,
        image_pe = image_pe,
        sparse_prompt_embeddings = sparse_noprompt,
        dense_prompt_embeddings = dense_noprompt,
        inter_feature = None,
        multimask_output = True
    )

    fig = plt.figure(figsize=(8,20))
    image = cv2.imread(vis_png)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax = fig.add_subplot(521)
    ax.imshow(image)
    ax.set_title('image')
    ax = fig.add_subplot(522)
    gt_color_mask = get_color_inst_mask(gt_inst_mask)
    ax.imshow(gt_color_mask)
    ax.set_title('gt instance mask')
    
    i = 3
    for outputs in [point_outputs, bbox_outputs, mask_outputs, noprompt_outputs]:
        pred_logits = outputs['logits_256']  # (bs, 4, 256, 256)
        pred_mask = (pred_logits[0,1,:,:] > 0).cpu().numpy()
        sam_img_embed_256 = outputs['upscaled_embedding'][0]  # (32, 256, 256)
        ax = fig.add_subplot(5,2,i)
        if i == 3:
            show_points(np.array([point//2]), np.array([1]), ax, marker_size=200)
        if i == 5:
            show_box(bbox//2, ax)
        embed_256 = torch.mean(sam_img_embed_256, dim=0).detach().cpu().numpy()
        ax.imshow(embed_256, cmap='hot')
        # if i == 7:
        #     show_mask(mask_prompt[0,0,...].cpu().numpy()>0, ax)
        i += 1
        ax = fig.add_subplot(5,2,i)
        ax.imshow(pred_mask, cmap='gray')
        i += 1
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{image_name}')
    plt.close()


if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    
    main()
