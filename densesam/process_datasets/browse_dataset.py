# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import argparse
from mmengine.registry import init_default_scope
from datasets.create_loader import gene_loader_eval
from mmdet.models.utils import mask2ndarray
from mmdet.structures.bbox import BaseBoxes
from mmengine.config import Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
from utils import show_multi_mask,show_box
from mmdet.structures import DetDataSample

parser = argparse.ArgumentParser()

# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vis_nums', type=int, default=-1)
parser.add_argument('--show_bbox', action='store_true')

def draw_dataset_gt(img, data_sample: DetDataSample, save_path, metainfo, show_bbox = True):
    classes = metainfo['classes']
    palette = metainfo['palette']

    gt_mask = data_sample.gt_sem_seg.sem_seg
    gt_boxes = data_sample.gt_instances.bboxes
    boxes_clsids = data_sample.gt_instances.labels

    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    show_multi_mask(gt_mask, ax, palette)
    if show_bbox:
        for box,cls_id in zip(gt_boxes, boxes_clsids): 
            cls_color = palette[cls_id]
            edgecolor = np.array([cls_color[0]/255, cls_color[1]/255, cls_color[2]/255, 1])
            show_box(box, ax, edgecolor=edgecolor)
    ax.set_title('GT info')
    patches = [mpatches.Patch(facecolor=np.array(palette[i])/255., label=classes[i], edgecolor='black') for i in range(len(classes))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # register all modules in mmdet into the registries
    init_default_scope('mmdet')
    d_cfg = Config.fromfile(args.dataset_config_file)
    dataloader, metainfo, restinfo = gene_loader_eval(
        dataset_config = d_cfg, seed = args.seed)
    for i_batch, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):

        if args.vis_nums > 0 and i_batch > args.vis_nums:
            break
        img = sampled_batch['inputs'][0].permute(1, 2, 0).int().numpy()
        img = img[..., [2, 1, 0]]  # bgr to rgb

        data_sample = sampled_batch['data_samples'][0].numpy()
        filename = data_sample.img_path.split('/')[-1]
        gt_instances = data_sample.gt_instances
        gt_bboxes = gt_instances.get('bboxes', None)
        if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
            gt_instances.bboxes = gt_bboxes.tensor
        gt_masks = gt_instances.get('masks', None)
        if gt_masks is not None:
            masks = mask2ndarray(gt_masks)
            gt_instances.masks = masks.astype(bool)
        data_sample.gt_instances = gt_instances

        out_file = f'{args.save_dir}/{filename}'
        draw_dataset_gt(img, data_sample, out_file, metainfo, args.show_bbox)
        print(f'{filename} instances number: {len(gt_instances.bboxes)}')


if __name__ == '__main__':
    main()

'''
python process_datasets/browse_dataset.py \
    configs/datasets/cnseg.py \
    visual_results/gt_visual/cnseg \
    --vis_nums 5 \
    --show_bbox
'''