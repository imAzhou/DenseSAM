import os
import time
import torch
from densesam.utils import set_seed
from densesam.models.sam.build_sam import sam_model_registry
from densesam.datasets.create_loader import gene_loader_trainval
from mmengine.config import Config
from tqdm import tqdm
import cv2
import numpy as np

seed = 1234
device = 'cuda:0'
dataset_config_file = 'densesam/configs/datasets/lip.py'
sam_ckpt_path = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth'
save_inners = [1]


if __name__ == "__main__":
    device = torch.device(device)
    set_seed(seed)
    d_cfg = Config.fromfile(dataset_config_file)
    
    # load datasets
    train_dataloader, val_dataloader, metainfo,_ = gene_loader_trainval(
        dataset_config = d_cfg, seed = seed)

    # register model
    sam_model = sam_model_registry['vit_h'](checkpoint = sam_ckpt_path).to(device)
    sam_model.eval()
    
    start = time.time()
    for dataloader in [train_dataloader, val_dataloader]:
        for i_batch, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):
            image_tensor = torch.stack(sampled_batch['inputs']).to(device)    # (bs, 3, 1024, 1024), 3 is bgr
            image_tensor_rgb = image_tensor[:, [2, 1, 0], :, :]

            # test_img = image_tensor_rgb[0].permute(1,2,0).contiguous().cpu().numpy()
            # cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR, test_img)
            # cv2.imwrite('cv2.jpg', test_img)

            with torch.no_grad():
                # input_images.shape: [bs, 3, 1024, 1024]
                input_images = sam_model.preprocess(image_tensor_rgb)
                # bs_image_embedding.shape: [bs, c=256, h=64, w=64]
                bs_image_embedding, bs_inner_feats = sam_model.image_encoder(input_images, need_inter=True)
                bs_inner_feats = torch.stack(bs_inner_feats,dim=0).permute(1,0,2,3,4)
            
            for datainfo, img_t, img_t_in in zip(sampled_batch['data_samples'], bs_image_embedding, bs_inner_feats):
                # /x22201018/datasets/MedicalDatasets/MoNuSeg/train/img_dir/TCGA-G9-6356-01Z-00-DX1.png
                img_path = datainfo.img_path
                image_name = img_path.split('/')[-1]
                pure_name = image_name.split('.')[0]
                img_emb_tensor = img_t.cpu().clone()
                img_emb_tensor_inner = img_t_in.cpu().clone()
                tensor_save_dir = '/'.join(img_path.split('/')[:-2])
                tensor_save_dir += '/img_tensor'
                os.makedirs(tensor_save_dir, exist_ok=True)
                torch.save(img_emb_tensor, f'{tensor_save_dir}/{pure_name}.pt')
                for idx in save_inners:
                    inner = img_emb_tensor_inner[idx]
                    torch.save(inner, f'{tensor_save_dir}/{pure_name}_inner_{idx}.pt')


    end = time.time()
    m,s = (end-start)//60,(end-start)%60
    print(f'cost time: {int(m)}m{int(s)}s')
    