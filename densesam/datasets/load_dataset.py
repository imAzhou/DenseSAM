import torch
import os
from mmdet.datasets import CocoPanopticDataset


class LoadDataset(CocoPanopticDataset):

       def __init__(self,**args) -> None:
            self.load_embed = args['load_embed']
            del args['load_embed']
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
            pipeline_data_info['gt_foreground_masks'] = pipeline_data_info['data_samples'].gt_sem_seg.sem_seg[0]
            
            if self.load_embed:
                img_path = data_info['img_path']
                img_dir,img_name = os.path.dirname(img_path),os.path.basename(img_path)
                purename = img_name.split('.')[0]
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
                
                prestore_inner_embed_path = f'{root_path}/{purename}_inner_1.pt'
                inter_feat = torch.load(prestore_inner_embed_path)
                pipeline_data_info['inter_feat'] = inter_feat

            return pipeline_data_info
       