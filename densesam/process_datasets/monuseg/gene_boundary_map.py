import os
import torch
import numpy as np
from mmengine.fileio import get
import mmcv
from panopticapi.utils import rgb2id
from scipy.ndimage import binary_dilation, binary_erosion
import cv2
from tqdm import tqdm

def onehot2instmask(onehot_mask, with_boundary=False):
    h,w = onehot_mask.shape[1],onehot_mask.shape[2]
    instmask = torch.zeros((h,w), dtype=torch.int32)
    inst_boundary_mask = np.zeros((len(onehot_mask), h, w))
    for iid,seg in enumerate(onehot_mask):
        instmask[seg.astype(bool)] = iid+1
        if with_boundary:
            dilated_mask = binary_dilation(seg)
            eroded_mask = binary_erosion(seg)
            boundary_mask = dilated_mask ^ eroded_mask   # binary mask (h,w)
            inst_boundary_mask[iid] = boundary_mask
    
    if with_boundary:
        inst_overlap_boundary_mask = np.sum(inst_boundary_mask, axis=0)
        inst_overlap_boundary_mask = (inst_overlap_boundary_mask > 1).astype(int)
        inst_overlap_boundary_mask = binary_dilation(inst_overlap_boundary_mask, iterations=2)
        inst_boundary_mask = np.max(inst_boundary_mask, axis=0)
        inst_boundary_mask[inst_overlap_boundary_mask] = 1
        return instmask, torch.as_tensor(inst_boundary_mask)
    
    return instmask


if __name__ == '__main__':
    BACKGROUND = 255
    
    root_dir = '/x22201018/datasets/MedicalDatasets/MoNuSeg'
    
    for mode in ['train_p256', 'test_p256']:
        boundary_save_dir = f'{root_dir}/{mode}/boundary_dir'
        os.makedirs(boundary_save_dir, exist_ok=True)

        for filename in tqdm(os.listdir(f'{root_dir}/{mode}/panoptic_seg_anns_coco')):
            seg_map_path = f'{root_dir}/{mode}/panoptic_seg_anns_coco/{filename}'
            img_bytes = get(seg_map_path)
            pan_png = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb').squeeze()
            background_seg = np.zeros_like(pan_png) + BACKGROUND
            background_id = np.unique(rgb2id(background_seg))[0]
            pan_ids = rgb2id(pan_png)

            inst_boundary_mask = np.zeros_like(pan_ids)

            for idx in np.unique(pan_ids):
                if idx != background_id:
                    instmask = (pan_ids == idx).astype(int)

                    dilated_mask = binary_dilation(instmask)
                    eroded_mask = binary_erosion(instmask)
                    boundary_mask = dilated_mask ^ eroded_mask   # binary mask (h,w)
                    # boundary_mask = binary_dilation(boundary_mask.astype(int), iterations=2)
                    # inst_boundary_mask[boundary_mask>0] = 255
                    inst_boundary_mask[boundary_mask] = 255
            cv2.imwrite(seg_map_path.replace('panoptic_seg_anns_coco','boundary_dir'), inst_boundary_mask)
            