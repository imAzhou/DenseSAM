import numpy as np
import os
from sklearn.model_selection import train_test_split
import tifffile
import pandas as pd
import cv2
import json
from tqdm import tqdm

root = '/x22201018/datasets/MedicalDatasets/CoNIC'
modes = ['train', 'val', 'test']
BACKGROUND_ID = 1
CELL_ID = 2
palette=[[255, 255, 255], [47, 243, 15]]
categories_info = [
    {'id': BACKGROUND_ID, 'name': 'Background', 'isthing': 0, 'color':palette[0], 'supercategory': '' },
    {'id': CELL_ID, 'name': 'Cell', 'isthing': 1, 'color':palette[1], 'supercategory': '' }
]

def create_ann():
    image_path = f'{root}/images.npy'
    mask_path = f'{root}/labels.npy'
    images = np.load(image_path)    # shape: (total_img_nums, h, w, c)
    masks = np.load(mask_path)      # shape: (total_img_nums, h, w, 2) 0:instanceid, 1: class id

    df = pd.read_csv(f'{root}/patch_info.csv')
    all_names = df.values

    total_idxs = range(len(images))

    train_idxs, temp_idxs = train_test_split(total_idxs, test_size=0.3, random_state=42)
    test_idxs, val_idxs = train_test_split(temp_idxs, test_size=2/3, random_state=42)

    split_idxs = [train_idxs, val_idxs, test_idxs]

    for mode, idxs_set in zip(modes, split_idxs):
        save_root_dir = f'{root}/{mode}'
        save_img_dir = f'{save_root_dir}/img_dir'
        os.makedirs(save_img_dir, exist_ok=True)
        
        panoptic_seg_anns = f'{save_root_dir}/panoptic_seg_anns'
        os.makedirs(panoptic_seg_anns, exist_ok=True)

        init_ann_json = {
            'categories': categories_info
        }

        all_imgs_info = []
        current_img_id = 0
        
        for idx in tqdm(idxs_set, ncols=70):
            img = images[idx]
            img_pure_name = all_names[idx][0].replace('-','_')
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{save_img_dir}/{img_pure_name}.png', img_bgr)
            inst_mask = masks[idx][:,:,0]   # (h,w)
            pan_iids = np.unique(inst_mask)
            h,w = inst_mask.shape

            all_imgs_info.append({
                'file_name': img_pure_name + '.png',
                'height': h,
                'width': w,
                'id': current_img_id,
            })

            # all pixes default background
            panoptic_ann = np.zeros((h, w, 3), dtype=np.int16)
            panoptic_ann[:, :, 0] = BACKGROUND_ID
            for iid in pan_iids[1:]:
                mask = inst_mask == iid
                panoptic_ann[mask] = (CELL_ID, iid, 0)
            tif_filename = img_pure_name + '.tif'
            ann_save_path = f'{panoptic_seg_anns}/{tif_filename}'
            tifffile.imwrite(ann_save_path, panoptic_ann)

            current_img_id += 1

        init_ann_json['images'] = all_imgs_info
        json_save_path = f'{root}/{mode}/panoptic_anns.json'
        with open(json_save_path, 'w', encoding='utf-8') as ann_f:
            json.dump(init_ann_json, ann_f)



if __name__ == '__main__':
    create_ann()