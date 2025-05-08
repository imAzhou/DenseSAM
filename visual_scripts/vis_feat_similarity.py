import torch
import os
import argparse
from utils import set_seed,show_points,show_box,show_mask
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
        input_images = model.preprocess(input_tensor).to(device)
        bs_image_embedding,inter_features = model.image_encoder(input_images, need_inter=True)
    
    return bs_image_embedding,inter_features

def get_similarity(featmap, point):
    target_vector = featmap[point[1], point[0], :]
    vector_norm = target_vector / target_vector.norm(dim=0, keepdim=True)  # (C,)
    featmap_norm = featmap / featmap.norm(dim=0, keepdim=True)  # (H, W, C)
    cosine_similarity = torch.sum(featmap_norm * vector_norm[None, None, :], dim=-1)  # (H, W)
    return cosine_similarity

def get_sam_embedings(img_path, points, device):
    sam_ckpt = 'checkpoints_sam/sam_vit_h_4b8939.pth'
    point_prompt = (
        torch.as_tensor(points, dtype=torch.float, device=device).unsqueeze(0),
        torch.ones(1, dtype=torch.int, device=device).unsqueeze(0)
    )

    sam_model = sam_model_registry['vit_h'](checkpoint = sam_ckpt).to(device)
    bs_image_embedding,inter_features = get_img_embed(img_path, sam_model, device)
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
        multimask_output = False
    )

    decoder_inner_embeds = point_outputs[-1]['decoder_inner_embeds']
    upscaled_embedding = point_outputs[-1]['upscaled_embedding']    # (bs,c,h,w)

    return inter_features, decoder_inner_embeds, upscaled_embedding,point_outputs

def get_ours_embedings(img_path, device):
    
    cfg = Config.fromfile('logs/cpm17/config.py')
    model = DenseSAMNet(
        use_local = cfg.use_local,
        use_global = cfg.use_global,
        use_boundary_head = cfg.use_boundary_head,
        use_embed = cfg.dataset.load_embed,
        sam_ckpt = cfg.sam_ckpt,
        sam_type = cfg.sam_type,
        device = device,
        use_inner_idx = cfg.use_inner_idx
    ).to(device)

    model.load_parameters('logs/cpm17/checkpoints/best.pth')
    
    bs_image_embedding,inter_features = get_img_embed(img_path, model, device)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points = None,
        boxes = None,
        masks = None,
    )
        
    image_pe = model.prompt_encoder.get_dense_pe()
    # low_res_masks.shape: (bs, num_cls, 256, 256)
    decoder_outputs = model.mask_decoder(
        image_embeddings = bs_image_embedding,
        image_pe = image_pe,
        sparse_prompt_embeddings = sparse_embeddings,
        dense_prompt_embeddings = dense_embeddings,
        inner_feats = inter_features
    )

    decoder_inner_embeds = decoder_outputs['decoder_inner_embeds']
    upscaled_embedding = decoder_outputs['upscaled_embedding']
    return inter_features, decoder_inner_embeds, upscaled_embedding,decoder_outputs

def vis_heatmap(fig,pos,img_embed,point_in_feat,tag_str):
    ax = fig.add_subplot(pos)
    embed_var = torch.var(img_embed, dim=-1).detach().cpu().numpy()
    ax.imshow(embed_var, cmap='hot')
    ax.set_title(f'variance in {tag_str}')

    ax = fig.add_subplot(pos+1)
    emb_sm = get_similarity(img_embed, point_in_feat)
    emb_sm = emb_sm.detach().cpu().numpy()
    ax.imshow(emb_sm, cmap='hot')
    show_points(np.array([point_in_feat]), np.array([1]), ax, marker_size=200)
    ax.set_title(f'similarity in {tag_str}')

def main():
    
    save_dir = f'visual_results/embed_similarity'
    os.makedirs(save_dir, exist_ok=True)
    root_dir = 'datasets/MedicalDatasets/CPM17/test_p512'
    # image_name = 'image_03_0'
    image_name = 'image_00_3'
    vis_png = f'{root_dir}/img_dir/{image_name}.png'
    gt_mask = f'{root_dir}/panoptic_seg_anns_coco/{image_name}.png'
    chosen_point = np.array([411,140])    # x,y

    img_mask = cv2.imread(gt_mask)
    h,w = img_mask.shape[:2]
    scale = 1024//h
    points = np.array([chosen_point*scale])

    # register model
    # inter_features, decoder_inner_embeds, upscaled_embedding, point_outputs = get_sam_embedings(vis_png, points, device)
    inter_features, decoder_inner_embeds, upscaled_embedding, decoder_outputs = get_ours_embedings(vis_png, device)

    # output_mask = point_outputs[0][0,0,...].detach().cpu().numpy()
    # output_mask = output_mask > 0

    output_mask = decoder_outputs['logits_256'][0,0,...].detach().cpu().numpy()
    output_mask = output_mask > 0
    output_boundary = decoder_outputs['logits_256'][0,1,...].detach().cpu().numpy()
    output_boundary = output_boundary > 0
    ESI_output = decoder_outputs['fused_feat'].view(1,64,64,-1)  # bs, h*w, c
    
    point_in_feat64 = chosen_point // (h // 64)
    point_in_feat256 = chosen_point // (h // 256)
    
    row,column = 3,2
    fig = plt.figure(figsize=(4*column,4*row))
    # image = cv2.imread(vis_png)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # ax = fig.add_subplot(row,column,1)
    # ax.imshow(image)
    # ax.set_title('image')

    # ax = fig.add_subplot(row,column,2)
    # ax.imshow(output_mask, cmap='gray')
    # ax.set_title('output mask')

    # ax = fig.add_subplot(row,column,3)
    # ax.imshow(output_boundary, cmap='gray')
    # ax.set_title('output boundary')

    
    vis_heatmap(fig, int(f'{row}{column}1'), ESI_output[0], point_in_feat64, 'fused feat')
    # vis_heatmap(fig, int(f'{row}{column}1'), decoder_inner_embeds[0][0], point_in_feat64, 'fused feat')
    # vis_heatmap(fig, int(f'{row}{column}3'), decoder_inner_embeds[0][0], point_in_feat64, '1st cross attn')
    vis_heatmap(fig, int(f'{row}{column}3'), decoder_inner_embeds[1][0], point_in_feat64, '2nd cross attn')
    vis_heatmap(fig, int(f'{row}{column}5'), upscaled_embedding[0].permute(1,2,0), point_in_feat256, 'upscaled')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/our_output_{image_name}.png')
    plt.close()


if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    
    main()
