import torch
import torch.nn as nn
from densesam.models.sam.build_sam import sam_model_registry
from .mask_decoder import MaskDecoder
from .transformer import TwoWayTransformer
import torch.nn.functional as F
import cv2
from torchvision import transforms as T
import numpy as np
from scipy import  ndimage
from skimage.segmentation import watershed

class DenseSAMNet(nn.Module):

    def __init__(self, 
                 sm_depth = 1,
                 use_inner_feat = False,
                 use_embed = False,
                 use_boundary_head = False,
                 sam_ckpt = None,
                 sam_type = 'vit_h',
                 device = None,
                 inter_idx = 1
                ):
        '''
        Args:
            - sm_depth: the depth of semantic module in transformer, 0 means don't use this module.
            - use_inner_feat: use inner featature or not
            - use_embed: if prestored image embedding, set this to true
            - use_boundary_head: dual-head for instance
            - sam_ckpt: sam weight path
            - sam_type: sam encoder type
        '''
        super(DenseSAMNet, self).__init__()
        assert sam_type in ['vit_b','vit_l','vit_h'], "sam_type must be in ['vit_b','vit_l','vit_h']"

        self.use_inner_feat = use_inner_feat
        self.inter_idx = inter_idx
        self.use_embed = use_embed
        self.device = device
        
        sam = sam_model_registry[sam_type](checkpoint = sam_ckpt)
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.preprocess = sam.preprocess
        self.postprocess_masks = sam.postprocess_masks

        encoder_embed_dim = {
            'vit_b': 768,
            'vit_l': 1024,
            'vit_h': 1280
        }
        self.mask_decoder = MaskDecoder(
            encoder_embed_dim = encoder_embed_dim[sam_type],
            transformer = TwoWayTransformer(
                sm_depth = sm_depth,
            ),
            use_inner_feat = use_inner_feat,
            use_boundary_head = use_boundary_head,
        )

        self.load_sam_parameters(sam.mask_decoder.state_dict())
        self.freeze_parameters()

    def load_sam_parameters(self, sam_mask_decoder_params: dict):
        
        self_mask_decoder_parmas = self.mask_decoder.state_dict()

        load_dict = {}
        load_params_from_sam = [
            'transformer', 'output_upscaling', 
        ]
        for name, param in sam_mask_decoder_params.items():
            for key in load_params_from_sam:
                if key in name:
                    load_dict[name] = param
                    break
        self_mask_decoder_parmas.update(load_dict)

        print('='*10 + ' load parameters from sam ' + '='*10)
        print(self.mask_decoder.load_state_dict(self_mask_decoder_parmas, strict = False))
        print('='*59)

    def freeze_parameters(self):

        update_param = [
            'semantic_module',
            'process_inter_feat',
            'mask_tokens',
            'output_hypernetworks_mlps',
            'output_boundary_mlps'
        ]

        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.prompt_encoder.named_parameters():
            param.requires_grad = False
        
        # freeze transformer
        for name, param in self.mask_decoder.named_parameters():
            need_update = False
            for key in update_param:
                if key in name:
                    need_update = True
                    break
            param.requires_grad = need_update
            
            # param.requires_grad = True
            
    def save_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        merged_dict = {
            'mask_decoder': self.mask_decoder.state_dict()
        }
        torch.save(merged_dict, filename)

    def load_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)
        print('='*10 + ' load parameters for mask decoder ' + '='*10)
        print(self.mask_decoder.load_state_dict(state_dict['mask_decoder'], strict = False))
        print('='*59)

    def forward(self, sampled_batch):
        
        bs_image_embedding,inter_feature = self.gene_img_embed(sampled_batch)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points = None,
            boxes = None,
            masks = None,
        )
        
        image_pe = self.prompt_encoder.get_dense_pe()
        # low_res_masks.shape: (bs, num_cls, 256, 256)
        decoder_outputs = self.mask_decoder(
            image_embeddings = bs_image_embedding,
            image_pe = image_pe,
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings = dense_embeddings,
            inter_feature = inter_feature,
        )
        # origin_size = sampled_batch['data_samples'][0].ori_shape
        # logits_1024 = F.interpolate(
        #     decoder_outputs['logits_256'],
        #     (1024,1024),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        # decoder_outputs['logits_1024'] = logits_1024
        
        return decoder_outputs
    
    def postprocess(self, datainfo, pred_result):
        pred_mask, pred_boundary = pred_result    # (1, h, w)
        (h,w) = datainfo.ori_shape   # (h,w)
        
        pred_mask = T.ToPILImage()(pred_mask.cpu()).convert('RGB')
        pred_mask = cv2.cvtColor(np.asarray(pred_mask),cv2.COLOR_RGB2GRAY)
        _,pred_mask = cv2.threshold(pred_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        pred_mask = cv2.resize(pred_mask, (w,h))
        _,pred_mask = cv2.threshold(pred_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
        pred_boundary = T.ToPILImage()(pred_boundary.cpu()).convert('RGB')
        pred_boundary = cv2.cvtColor(np.asarray(pred_boundary),cv2.COLOR_RGB2GRAY)  
        pred_boundary = cv2.normalize(pred_boundary, dst=None, alpha=350, beta=10, norm_type=cv2.NORM_MINMAX)
        _,pred_boundary = cv2.threshold(pred_boundary, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        pred_boundary = cv2.resize(pred_boundary, (w,h))

        pred_contours = cv2.bitwise_and(pred_boundary, pred_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3)) 
        pred_contours = cv2.erode(pred_contours, kernel, iterations=1)
        pred_contours = cv2.dilate(pred_contours, kernel, iterations=1)
        contourpoint, hierarchy = cv2.findContours(pred_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imagep = np.zeros([h, w], dtype=np.int32)
        sumArea=[cv2.contourArea(points) for points in contourpoint]
        arr_mean = np.mean(sumArea)
        minarea = arr_mean / 5
        for i in range(0,len(contourpoint)):
            if cv2.contourArea(contourpoint[i]) > minarea:
                imagep = cv2.drawContours(imagep, contourpoint, i, 255, -1)
        marker = ndimage.label(imagep)[0]
        # pred_instmask: 0: background, 1-N: instance_id
        pred_instmask = watershed(pred_mask, markers=marker, mask=pred_mask, watershed_line=True)

        return pred_instmask, pred_boundary!=255
    
    def gene_img_embed(self, sampled_batch: dict):
        
        if self.use_embed:
            bs_image_embedding = torch.stack(sampled_batch['img_embed']).to(self.device)
            if self.use_inner_feat:
                inter_feature = torch.stack(sampled_batch['inter_feat']).to(self.device)
            else:
                inter_feature = None
        else:
            with torch.no_grad():
                # input_images.shape: [bs, 3, 1024, 1024]
                image_tensor = torch.stack(sampled_batch['inputs'])  # (bs, 3, 1024, 1024), 3 is bgr
                # image_tensor_rgb = image_tensor[:, [2, 1, 0], :, :]
                # input_images = self.preprocess(image_tensor_rgb).to(self.device)
                input_images = image_tensor.to(self.device)
                # bs_image_embedding.shape: [bs, c=256, h=64, w=64]
                if self.use_inner_feat:
                    bs_image_embedding,inter_feature = self.image_encoder(input_images, need_inter=True)
                    bs_image_embedding,inter_feature = bs_image_embedding.detach(),inter_feature[self.inter_idx].detach()
                else:
                    bs_image_embedding = self.image_encoder(input_images, need_inter=False).detach()
                    inter_feature = None

        return bs_image_embedding,inter_feature
    