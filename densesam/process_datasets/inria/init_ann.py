import cv2
import os
import numpy as np
import tifffile
import json
from tqdm import tqdm
import shutil


data_root = '/x22201018/datasets/RemoteSensingDatasets/InriaBuildingDataset'
modes = ['train_new','val_new']
BACKGROUND_ID = 1
FOREGROUND_ID = 2
palette=[[255, 255, 255], [244, 251, 4]]
categories_info = [
    {'id': BACKGROUND_ID, 'name': 'Background', 'isthing': 0, 'color':palette[0], 'supercategory': '' },
    {'id': FOREGROUND_ID, 'name': 'Building', 'isthing': 1, 'color':palette[1], 'supercategory': '' }
]

def create_coco_init():

    for mode in modes:
        src_img_dir = f'{data_root}/{mode}/img_dir'
        mask_img_dir = f'{data_root}/{mode}/ann_dir'
        
        target_dir = f'{data_root}/{mode}'
        # target_img_dir = f'{target_dir}/img_dir'
        target_ann_mask_dir = f'{target_dir}/panoptic_seg_anns'
        # os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_ann_mask_dir, exist_ok=True)

        init_ann_json = {
            'categories': categories_info
        }
        all_imgs_info = []
        current_img_id = 0

        all_imgs = os.listdir(src_img_dir)
        for filename in tqdm(all_imgs):
            # shutil.move(f'{src_img_dir}/{filename}', f'{target_img_dir}/{filename}')
            mask_path = f'{mask_img_dir}/{filename}'
            image_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            h,w = image_mask.shape
            panoptic_ann = np.zeros((h, w, 3), dtype=np.int16)
            panoptic_ann[:, :, 0] = BACKGROUND_ID
            panoptic_ann[image_mask == 255] = [FOREGROUND_ID, 1, 0]

            purename = filename.split('.')[0]
            save_inst_mask_path = f'{target_ann_mask_dir}/{purename}.tif'
            tifffile.imwrite(save_inst_mask_path, panoptic_ann)
            all_imgs_info.append({
                'file_name': filename,
                'height': h,
                'width': w,
                'id': current_img_id,
            })
            current_img_id += 1
        
        init_ann_json['images'] = all_imgs_info
        json_save_path = f'{target_dir}/panoptic_anns.json'
        with open(json_save_path, 'w', encoding='utf-8') as ann_f:
            json.dump(init_ann_json, ann_f)
            

if __name__ == '__main__':
    create_coco_init()
