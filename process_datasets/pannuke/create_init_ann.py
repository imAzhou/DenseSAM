'''
See details in https://github.com/cocodataset/panopticapi

This file function:
    1. generate panoptic annotation json file, which format like:
        ann_json = dict(
            images = [{
                'file_name': '000000001268.jpg',
                'height': 427,
                'width': 640,
                'id': 1268,
            }],
            categories = [{
                {'id': 0, 'name': 'person', 'isthing': 0 or 1, 'color':[R,G,B], 'supercategory': 'cell' }
            }]
        )
    2. generate panoptic annotation mask png, which have 2 channels, channel 0 means semantic category label, channel 1 means instance ID label, stored by np.uint32.
'''

import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import tifffile

root = '/x22201018/datasets/MedicalDatasets/PanNuke'
parts = ['Part1', 'Part2', 'Part3']

def init_binary_cls_anno():

    palette=[[255, 255, 255], [47, 243, 15]]
    categories_info = [
        {'id': 1, 'name': 'Background', 'isthing': 0, 'color':palette[0], 'supercategory': '' },
        {'id': 2, 'name': 'Cell', 'isthing': 1, 'color':palette[1], 'supercategory': '' }
    ]

    for part in parts:
        masks_dict_path = os.path.join(root, part, 'Masks', 'masks.json')
        mask_path = os.path.join(root, part, 'Masks', 'masks.npy')
        masks = np.load(mask_path)      # shape: (total_img_nums, h, w, cls_num)

        panoptic_seg_anns = f'{root}/{part}/panoptic_binary_seg_anns'
        os.makedirs(panoptic_seg_anns, exist_ok=True)

        init_ann_json = {
            'categories': categories_info
        }

        with open(masks_dict_path,'r',encoding='utf-8') as f :
            masks_dict = json.load(f)
            all_imgs_info = []
            current_img_id = 0
            for filename,mask_idx in tqdm(masks_dict.items(), ncols=70):
                mask_ann = masks[mask_idx]  # shape: (h, w, cls_num)
                h,w = mask_ann.shape[0], mask_ann.shape[1]

                all_imgs_info.append({
                    'file_name': filename,
                    'height': h,
                    'width': w,
                    'id': current_img_id,
                })

                # 0 is ignore pixel value in panoptic_ann_png
                panoptic_ann_png = np.zeros((h, w, 3), dtype=np.uint16)
                idx = 1
                for cls_id in range(6): # 0-4: cell, 5: background
                    cls_mask = mask_ann[...,cls_id] # shape: (h, w)
                    instance_ids = np.unique(cls_mask)
                    instance_ids = instance_ids[1:] # remove 0 from instance ids set

                    for iid in instance_ids:
                        classes_id = 1 if cls_id == 5 else 2
                        panoptic_ann_png[cls_mask == iid, 0] = classes_id
                        panoptic_ann_png[cls_mask == iid, 1] = idx
                        idx += 1
                tif_filename = filename.split('.')[0] + '.tif'
                ann_save_path = f'{panoptic_seg_anns}/{tif_filename}'
                tifffile.imwrite(ann_save_path, panoptic_ann_png)
                # cv2.imwrite(ann_save_path, panoptic_ann_png)
                current_img_id += 1

        init_ann_json['images'] = all_imgs_info
        json_save_path = f'{root}/{part}/panoptic_binary_anns.json'
        with open(json_save_path, 'w', encoding='utf-8') as ann_f:
            json.dump(init_ann_json, ann_f)

if __name__ == '__main__':
    init_binary_cls_anno()