import torch
import os
import argparse
from densesam.utils import set_seed,show_points,show_box,show_mask
from densesam.models.sam.build_sam import sam_model_registry
import matplotlib.pyplot as plt
import cv2
import numpy as np
from densesam.models.dense_sam import DenseSAMNet
from mmengine.config import Config
import torch.nn.functional as F

parser = argparse.ArgumentParser()

# base args
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

def get_img_embed(img_path, model, device):
    img = cv2.imread(img_path)
    input_img = cv2.resize(img, (1024,1024))
    input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
    input_tensor = torch.as_tensor(input_img).permute(2,0,1).unsqueeze(0)   # (1,C,H,W)

    with torch.no_grad():
        input_images = model.preprocess(input_tensor.to(device))
        bs_image_embedding,inter_features = model.image_encoder(input_images, need_inter=True)
    
    return bs_image_embedding,inter_features

def get_similarity(featmap, point):
    target_vector = featmap[point[1], point[0], :]
    vector_norm = target_vector / target_vector.norm(dim=0, keepdim=True)  # (C,)
    featmap_norm = featmap / featmap.norm(dim=0, keepdim=True)  # (H, W, C)
    cosine_similarity = torch.sum(featmap_norm * vector_norm[None, None, :], dim=-1)  # (H, W)
    return cosine_similarity

def get_sam_embedings(img_path, device, location_prompt):
    sam_ckpt = 'checkpoints_sam/sam_vit_h_4b8939.pth'
    points_prompt,boxes_prompt, mask_prompt = location_prompt

    sam_model = sam_model_registry['vit_h'](checkpoint = sam_ckpt).to(device)
    bs_image_embedding,inter_features = get_img_embed(img_path, sam_model, device)
    image_pe = sam_model.prompt_encoder.get_dense_pe()
    sparse_point,dense_point = sam_model.prompt_encoder(
        points = points_prompt,
        boxes = boxes_prompt,
        masks = mask_prompt,
    )
    point_outputs = sam_model.mask_decoder(
        image_embeddings = bs_image_embedding,
        image_pe = image_pe,
        sparse_prompt_embeddings = sparse_point,
        dense_prompt_embeddings = dense_point,
        multimask_output = False
    )

    decoder_inner_embeds = point_outputs[-1]['decoder_inner_embeds']
    upscaled_embedding = point_outputs[-1]['upscaled_embedding']    # (bs,c,h,w)

    return inter_features, decoder_inner_embeds, upscaled_embedding,point_outputs


def vis_heatmap(fig,pos,img_embed,binary_mask,star_point,tag_str):

    ax = fig.add_subplot(pos)
    embed_var = torch.var(img_embed, dim=-1).detach().cpu().numpy()
    ax.imshow(embed_var, cmap='hot')
    ax.set_title(f'variance in {tag_str}')

    ax = fig.add_subplot(pos+1)
    emb_sm = get_similarity(img_embed, star_point)
    emb_sm = emb_sm.detach().cpu().numpy()
    ax.imshow(emb_sm, cmap='hot')
    show_points(np.array([star_point]), np.array([1]), ax, marker_size=200)
    ax.set_title(f'similarity in {tag_str}')

    ax = fig.add_subplot(pos+2)
    ax.imshow(binary_mask, cmap='gray')
    ax.set_title(f'output mask in {tag_str}')

def main():
    
    save_dir = f'visual_results/ordinary_analyze'
    os.makedirs(save_dir, exist_ok=True)
    # root_dir = 'datasets/MedicalDatasets/CPM17/test_p512'
    # image_name = 'image_03_0'
    # image_name = 'image_00_3'
    # vis_png = f'{root_dir}/img_dir/{image_name}.png'
    # gt_mask = f'{root_dir}/panoptic_seg_anns_coco/{image_name}.png'
    
    # chosen_point = np.array([411,140])    # x,y
    # mulit_chosen_points = np.array([[411,140],[68,205],[271,388],[191,269]])    # x,y
    # chosen_bbox = np.array([373,105,452,177])    # x1,y1,x2,y2

    image_name = 'JFSW_1_0_47.png'
    vis_png = f'visual_results/ordinary_analyze/{image_name}'
    img = cv2.imread(vis_png)
    h,w = img.shape[:2]
    scale = 1024/h
    '''
    3900: [290,156]
    4158: [153,213]
    5148: [270,241]
    4134: [317,264]
    '''
    chosen_point = np.array([469,182])    # x,y

    point_prompt = (
        torch.as_tensor(np.array([chosen_point*scale]), dtype=torch.float, device=device).unsqueeze(0),
        torch.as_tensor([1], dtype=torch.int, device=device).unsqueeze(0)
    )
    # multi_point_prompt = (
    #     torch.as_tensor(mulit_chosen_points*scale, dtype=torch.float, device=device).unsqueeze(0),
    #     torch.as_tensor([1,1,1,0], dtype=torch.int, device=device).unsqueeze(0)
    # )
    # bbox_prompt = torch.as_tensor(chosen_bbox[None, :]*scale, dtype=torch.float, device=device).unsqueeze(0)

    location_prompts = [
        (point_prompt,None,None)
        # (None,None,None),
        # (None,bbox_prompt,None),
        # (multi_point_prompt,None,None),
    ]

    row,column = 1,3
    fig = plt.figure(figsize=(4*column,4*row))
    # for lp,tag,i in zip(location_prompts,['Non-prompt','bbox','multi points'],[1]):
    for lp,tag,i in zip(location_prompts,['single points'],[1]):
        inter_features, decoder_inner_embeds, upscaled_embedding, point_outputs = get_sam_embedings(vis_png, device,lp)
        star_point = chosen_point / (h / 64)
        star_point = [round(star_point[0]), round(star_point[1])]
        binary_mask = (point_outputs[0] > 0).detach().cpu().numpy()[0][0]
        vis_heatmap(fig, int(f'{row}{column}{i}'), inter_features[-1][0], binary_mask, star_point, tag)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/sam_prompt_output_{image_name}.png')
    plt.close()


if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    
    main()
