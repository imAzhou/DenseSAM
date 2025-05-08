import torch
import os
import argparse
from utils import set_seed,show_points,show_box,show_mask
from densesam.models.sam.build_sam import sam_model_registry
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn as nn

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

def get_sam_embedings(img_path, device):
    sam_ckpt = 'checkpoints_sam/sam_vit_h_4b8939.pth'
    
    sam_model = sam_model_registry['vit_h'](checkpoint = sam_ckpt).to(device)
    bs_image_embedding,inter_features = get_img_embed(img_path, sam_model, device)
    image_pe = sam_model.prompt_encoder.get_dense_pe()
    sparse_point,dense_point = sam_model.prompt_encoder(
        points = None,
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

    return inter_features, decoder_inner_embeds, upscaled_embedding,bs_image_embedding


def main():
    
    save_dir = f'visual_results/feat_downsample'
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
    point_in_feat64 = chosen_point // (h // 64)
    point_in_feat32 = chosen_point // (h // 16)
    
    # register model
    inter_features, decoder_inner_embeds, upscaled_embedding,bs_image_embedding = get_sam_embedings(vis_png, device)

    maxpool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
    avgpool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
    maxoutput = maxpool(bs_image_embedding)    # 1, C, 32, 32
    avgoutput = avgpool(bs_image_embedding)    # 1, C, 32, 32
    
    row,column = 4,2
    fig = plt.figure(figsize=(4*column,4*row))
    image = cv2.imread(vis_png)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax = fig.add_subplot(row,column,1)
    ax.imshow(image)
    ax.set_title('image')
    
    ax = fig.add_subplot(row,column,3)
    img_embed = bs_image_embedding[0].permute(1,2,0)  # (64, 64, C)
    embed_var = torch.var(img_embed, dim=-1).detach().cpu().numpy()
    ax.imshow(embed_var, cmap='hot')
    ax.set_title('variance in encoder final output')

    ax = fig.add_subplot(row,column,4)
    emb_sm = get_similarity(img_embed, point_in_feat64)
    emb_sm = emb_sm.detach().cpu().numpy()
    ax.imshow(emb_sm, cmap='hot')
    show_points(np.array([point_in_feat64]), np.array([1]), ax, marker_size=200)
    ax.set_title('similarity in encoder final output')

    ax = fig.add_subplot(row,column,5)
    img_embed = maxoutput[0].permute(1,2,0)
    embed_var = torch.var(img_embed, dim=-1).detach().cpu().numpy()
    ax.imshow(embed_var, cmap='hot')
    ax.set_title('variance after max output')

    ax = fig.add_subplot(row,column,6)
    emb_sm = get_similarity(img_embed, point_in_feat32)
    emb_sm = emb_sm.detach().cpu().numpy()
    ax.imshow(emb_sm, cmap='hot')
    show_points(np.array([point_in_feat32]), np.array([1]), ax, marker_size=200)
    ax.set_title('similarity after max output')

    ax = fig.add_subplot(row,column,7)
    img_embed = avgoutput[0].permute(1,2,0)
    embed_var = torch.var(img_embed, dim=-1).detach().cpu().numpy()
    ax.imshow(embed_var, cmap='hot')
    ax.set_title('variance after avg output')

    ax = fig.add_subplot(row,column,8)
    emb_sm = get_similarity(img_embed, point_in_feat32)
    emb_sm = emb_sm.detach().cpu().numpy()
    ax.imshow(emb_sm, cmap='hot')
    show_points(np.array([point_in_feat32]), np.array([1]), ax, marker_size=200)
    ax.set_title('similarity after avg output')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/ws4_encoderoutput_{image_name}.png')
    plt.close()


if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    
    main()
