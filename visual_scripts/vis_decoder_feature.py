import os
import torch
import argparse
from utils import set_seed
from models.dense_sam import DenseSAMNet
from datasets.create_loader import gene_loader_trainval
from mmengine.config import Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser()

# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt_path', type=str, default='cuda:0')
parser.add_argument('--save_interval', type=int, default=20, help='random seed')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

def draw_embed(draw_items, datainfo, pred_save_dir):
    img_path = datainfo.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    embeds_mean = draw_items['embeds_mean']
    embeds_var = draw_items['embeds_var']
    columns = len(embeds_mean) + 1
    
    fig = plt.figure(figsize=(3*columns,8))
    ax = fig.add_subplot(2,columns,1)
    ax.imshow(img)
    ax.set_axis_off()

    for i in range(len(embeds_mean)):
        ax = fig.add_subplot(2,columns,i+2)
        ax.imshow(embeds_mean[i], cmap='hot')
        ax.set_title(f'inner embed {i} mean')
    
    for i in range(len(embeds_var)):
        ax = fig.add_subplot(2,columns,i+2+columns)
        ax.imshow(embeds_var[i], cmap='hot')
        ax.set_title(f'inner embed {i} var')
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()


def main():
    # load datasets
    train_dataloader, val_dataloader, metainfo, restinfo = gene_loader_trainval(
        dataset_config = cfg, seed = args.seed)
    
    dataset_tag = cfg['dataset_tag']
    specified_image_name = 'image_00_3.png'
    pred_save_dir = f'visual_results/our_output/{dataset_tag}'
    os.makedirs(pred_save_dir, exist_ok=True)

    # register model
    model = DenseSAMNet(
                sm_depth = cfg.semantic_module_depth,
                use_inner_feat = cfg.use_inner_feat,
                use_embed = cfg.dataset.load_embed,
                use_boundary_head = cfg.use_boundary_head,
                sam_ckpt = cfg.sam_ckpt,
                sam_type = cfg.sam_type,
                device = device
            ).to(device)
    
    model.eval()
    model.load_parameters(args.ckpt_path)
    for i_batch, sampled_batch in enumerate(tqdm(val_dataloader, ncols=70)):

        with torch.no_grad():
            outputs = model(sampled_batch)
            pred_logits = outputs['logits_256']
            img_embed_256 = outputs['upscaled_embedding']  # (bs, 32, 256, 256)

        data_sample = sampled_batch['data_samples'][0]
        img_path = data_sample.img_path
        image_name = img_path.split('/')[-1]
        if image_name != specified_image_name:
            continue
        
        embed_256_mean = torch.mean(img_embed_256[0], dim=0).detach().cpu().numpy()
        embed_256_var = torch.var(img_embed_256[0], dim=0).detach().cpu().numpy()
        draw_items = dict(
            embeds_mean = [embed_256_mean],
            embeds_var = [embed_256_var]
        )
        draw_embed(draw_items, data_sample, pred_save_dir)
            

if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    
    main()



'''
python visual_scripts/vis_decoder_feature.py \
    logs/cpm17/config.py \
    logs/cpm17/checkpoints/best.pth \
    --save_interval 10
'''
