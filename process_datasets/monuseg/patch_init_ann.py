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
import xml.etree.ElementTree as ET
import tifffile
from skimage import draw
import matplotlib.pyplot as plt
import cv2
from mmdet.structures.mask import mask2bbox
from utils import show_mask,show_box
import torch
import torch.nn.functional as F
from einops import rearrange

root = 'datasets/MedicalDatasets/MoNuSeg'
modes = ['train', 'val', 'test']
BACKGROUND_ID = 1
CELL_ID = 2
palette=[[255, 255, 255], [47, 243, 15]]
categories_info = [
    {'id': BACKGROUND_ID, 'name': 'Background', 'isthing': 0, 'color':palette[0], 'supercategory': '' },
    {'id': CELL_ID, 'name': 'Cell', 'isthing': 1, 'color':palette[1], 'supercategory': '' }
]

def parse_xml(xml_path: str):
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    all_instance = root.findall('.//Annotation/Regions/Region')
    
    regions = []
    for item in all_instance:
        inst_vertexes = item.find('Vertices')
        vertices = inst_vertexes.findall('Vertex')
        coords = np.zeros((len(vertices), 2))
        for i, vertex in enumerate(vertices):
            coords[i][0] = vertex.attrib["X"]
            coords[i][1] = vertex.attrib["Y"]
        regions.append(coords)

    return regions
        

def init_cls_anno(size, overlap):

    for mode in modes:
        img_dir = f'{root}/{mode}/img_dir'
        ann_dir = f'{root}/{mode}/annotations'
        
        save_root_dir = f'{root}/{mode}_p{size}'
        if overlap > 0:
            save_root_dir = f'{root}/{mode}_p{size}_o{overlap}'

        save_img_dir = f'{save_root_dir}/img_dir'
        os.makedirs(save_img_dir, exist_ok=True)
        
        panoptic_seg_anns = f'{save_root_dir}/panoptic_seg_anns'
        os.makedirs(panoptic_seg_anns, exist_ok=True)

        init_ann_json = {
            'categories': categories_info
        }

        all_imgs_info = []
        current_img_id = 0
        
        all_imgs = os.listdir(ann_dir)
        for filename in tqdm(all_imgs):
            pure_filename = filename.split('.')[0]
            ann_path = f'{ann_dir}/{pure_filename}.xml'
            img_path = f'{img_dir}/{pure_filename}.png'
            img = cv2.imread(img_path)
            h,w = img.shape[0], img.shape[1]
            regions = parse_xml(ann_path)
            # all pixes default background
            panoptic_ann = np.zeros((h, w, 3), dtype=np.int16)
            panoptic_ann[:, :, 0] = BACKGROUND_ID
            for iid,pts in enumerate(regions):
                vertex_row_coords = pts[:, 0]
                vertex_col_coords = pts[:, 1]
                fill_row_coords, fill_col_coords = draw.polygon(
                    vertex_col_coords, vertex_row_coords, (h,w)
                )
                panoptic_ann[fill_row_coords, fill_col_coords] = (CELL_ID, iid+1, 0)

            img_t = torch.as_tensor(img).permute(2,0,1).contiguous()    # (3,h,w)
            img_t = F.interpolate(img_t.float().unsqueeze(0), (1024, 1024), mode="bilinear").squeeze(0)
            
            ann_t = torch.as_tensor(panoptic_ann).permute(2,0,1).contiguous()    # (3,h,w)
            ann_t = F.interpolate(ann_t.float().unsqueeze(0), (1024, 1024), mode="nearest").squeeze(0)

            img_ann_t = torch.cat([img_t, ann_t], dim=0)    # (6,h,w)
            if overlap > 0:
                img_ann_t = img_ann_t.unfold(1, size, size - overlap).unfold(2, size, size - overlap)   # (6,k,k,size,size)
            else:
                img_ann_t = rearrange(img_ann_t, "c (h i) (w j) -> c h w i j", i=size, j=size)   # (6,k,k,size,size)
            
            img_ann_t = img_ann_t.flatten(1,2)   # (6,k*k,size,size)
            img_t = img_ann_t[:3, ...].permute(1,2,3,0).contiguous()   # (k*k,size,size,3)
            ann_t = img_ann_t[3:, ...].permute(1,2,3,0).contiguous()   # (k*k,size,size,3)

            patch_idxes = range(img_t.shape[0])
            for idx,img_patch,ann_patch in zip(patch_idxes, img_t, ann_t):
                img_patch = img_patch.numpy()
                cv2.imwrite(f'{save_img_dir}/{pure_filename}_{idx}.png', img_patch)
                ann_patch = ann_patch.numpy().astype(np.int16)
                tifffile.imwrite(f'{panoptic_seg_anns}/{pure_filename}_{idx}.tif', ann_patch)
                all_imgs_info.append({
                    'file_name': f'{pure_filename}_{idx}.png',
                    'height': size,
                    'width': size,
                    'id': current_img_id,
                })
                current_img_id += 1
        
        init_ann_json['images'] = all_imgs_info
        json_save_path = f'{save_root_dir}/panoptic_anns.json'
        with open(json_save_path, 'w', encoding='utf-8') as ann_f:
            json.dump(init_ann_json, ann_f)


if __name__ == '__main__':
    size = 256
    init_cls_anno(size, overlap=64)
    # init_cls_anno(size, overlap=64)
