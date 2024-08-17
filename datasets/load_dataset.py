import torch
import os
from mmdet.datasets import CocoPanopticDataset
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

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

class LoadDataset(CocoPanopticDataset):

       def __init__(self,**args) -> None:
            self.load_embed = args['load_embed']
            self.load_inst = args['load_inst']
            del args['load_embed']
            del args['load_inst']
            super(LoadDataset, self).__init__(**args)
              

       def prepare_data(self, idx):
            """Get data processed by ``self.pipeline``.

            Args:
            idx (int): The index of ``data_info``.

            Returns:
            Any: Depends on ``self.pipeline``.
            """
            data_info = self.get_data_info(idx)
            pipeline_data_info = self.pipeline(data_info)
            img_path = data_info['img_path']
            img_dir,img_name = os.path.dirname(img_path),os.path.basename(img_path)
            purename = img_name.split('.')[0]
            pipeline_data_info['gt_foreground_masks'] = pipeline_data_info['data_samples'].gt_sem_seg.sem_seg[0]

            if self.load_inst:
                gt_inst_seg = pipeline_data_info['data_samples'].gt_instances.masks.masks  # np.array, (inst_nums, h, w)
                gt_inst_mask,gt_boundary_mask = onehot2instmask(gt_inst_seg, with_boundary=True)
                pipeline_data_info['gt_inst_mask'] = gt_inst_mask
                pipeline_data_info['gt_boundary_mask'] = gt_boundary_mask
            
            if self.load_embed:
                root_path = img_dir.replace('img_dir', 'img_tensor')
                prestore_embed_path = f'{root_path}/{purename}.pt'
                image_embedding = torch.load(prestore_embed_path)
                pipeline_data_info['img_embed'] = image_embedding

                # all_inner_t = []
                # for i in [0]:
                #     prestore_inner_embed_path = f'{root_path}/{purename}_inner_{i}.pt'
                #     inter_feat = torch.load(prestore_inner_embed_path)
                #     all_inner_t.append(inter_feat)
                # all_inner_t = torch.stack(all_inner_t)
                # pipeline_data_info['inter_feat'] = all_inner_t
                
                prestore_inner_embed_path = f'{root_path}/{purename}_inner_0.pt'
                inter_feat = torch.load(prestore_inner_embed_path)
                pipeline_data_info['inter_feat'] = inter_feat
                   
            return pipeline_data_info
       