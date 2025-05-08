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
        

def init_binary_cls_anno():
    for mode in modes:
        img_dir = f'{root}/{mode}/img_dir'
        ann_dir = f'{root}/{mode}/annotations'

        panoptic_seg_anns = f'{root}/{mode}/panoptic_seg_anns'
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

            all_imgs_info.append({
                'file_name': f'{pure_filename}.png',
                'height': h,
                'width': w,
                'id': current_img_id,
            })

            regions = parse_xml(ann_path)
            # all pixes default background
            panoptic_ann_png = np.zeros((h, w, 3), dtype=np.uint16)
            panoptic_ann_png[:, :, 0] = BACKGROUND_ID

            for iid,pts in enumerate(regions):
                # cv2.fillPoly(panoptic_ann_png, np.array([pts]), (CELL_ID, iid+1, 0))
                vertex_row_coords = pts[:, 0]
                vertex_col_coords = pts[:, 1]
                fill_row_coords, fill_col_coords = draw.polygon(
                    vertex_col_coords, vertex_row_coords, (h,w)
                )
                panoptic_ann_png[fill_row_coords, fill_col_coords] = (CELL_ID, iid+1, 0)
            ann_save_path = f'{panoptic_seg_anns}/{pure_filename}.tif'
            tifffile.imwrite(ann_save_path, panoptic_ann_png)
            current_img_id += 1
        
        init_ann_json['images'] = all_imgs_info
        json_save_path = f'{root}/{mode}/panoptic_anns.json'
        with open(json_save_path, 'w', encoding='utf-8') as ann_f:
            json.dump(init_ann_json, ann_f)

def visual_gt_info():
    for mode in modes:
        img_dir = f'{root}/{mode}/img_dir'
        ann_dir = f'{root}/{mode}/annotations'

        vis_seg_anns = f'{root}/{mode}/gt_visual'
        os.makedirs(vis_seg_anns, exist_ok=True)

        all_imgs = os.listdir(ann_dir)
        for filename in tqdm(all_imgs):
            pure_filename = filename.split('.')[0]
            ann_path = f'{ann_dir}/{pure_filename}.xml'
            img_path = f'{img_dir}/{pure_filename}.png'
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h,w = img.shape[0], img.shape[1]

            gt_visual_filename = f'{vis_seg_anns}/{pure_filename}.png'

            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.set_title('gt info')

            regions = parse_xml(ann_path)
            binary_mask = torch.zeros((h, w), dtype=torch.int8)
            all_boxes = []
            for iid,pts in enumerate(regions):
                segment_mask = torch.zeros((h, w), dtype=torch.int8)
                vertex_row_coords = pts[:, 0]
                vertex_col_coords = pts[:, 1]
                fill_row_coords, fill_col_coords = draw.polygon(
                    vertex_col_coords, vertex_row_coords, (h,w)
                )
                binary_mask[fill_row_coords, fill_col_coords] = 1
                segment_mask[fill_row_coords, fill_col_coords] = 1
                bbox = mask2bbox(segment_mask.unsqueeze(0))[0]
                all_boxes.append(bbox)
            show_mask(binary_mask, ax, rgb = palette[1])
            edgecolor = np.array([palette[1][0]/255, palette[1][1]/255, palette[1][2]/255, 1])
            for bbox in all_boxes:
                show_box(bbox, ax, edgecolor = edgecolor)
            
            plt.tight_layout()
            plt.savefig(gt_visual_filename)
            plt.close()


if __name__ == '__main__':
    init_binary_cls_anno()
    # visual_gt_info()
