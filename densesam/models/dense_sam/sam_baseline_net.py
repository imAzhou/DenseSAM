import torch
import torch.nn as nn
from densesam.models.sam.build_sam import sam_model_registry
import torch.nn.functional as F

class SAMBaselineNet(nn.Module):

    def __init__(self,
                 use_embed = False,
                 update_decoder = False,
                 sam_ckpt = None,
                 sam_type = 'vit_h',
                 device = None,
                ):
        '''
        Args:
            - num_classes: determined mask_tokens num,
                -1: sam origin mask decoder, 1: binary segmantation, >1: multi class segmentation.
            - sm_depth: the depth of semantic module in transformer, 0 means don't use this module.
        '''
        super(SAMBaselineNet, self).__init__()
        assert sam_type in ['vit_b','vit_l','vit_h'], "sam_type must be in ['vit_b','vit_l','vit_h']"

        self.use_embed = use_embed
        self.device = device
        
        sam = sam_model_registry[sam_type](checkpoint = sam_ckpt)
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.preprocess = sam.preprocess

        self.freeze_parameters(update_decoder)

    def freeze_parameters(self, update_decoder):

        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.prompt_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.mask_decoder.named_parameters():
            param.requires_grad = update_decoder

            
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

    def forward(self, sampled_batch, prompt_dict):
        
        bs_image_embedding,inter_features = self.gene_img_embed(sampled_batch)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points = prompt_dict['point_prompt'],
            boxes = prompt_dict['box_prompt'],
            masks = None,
        )
        
        image_pe = self.prompt_encoder.get_dense_pe()
        # logits_256.shape: (bs, 1, 256, 256)
        logits_256, iou_pred, extra_outputs = self.mask_decoder(
            image_embeddings = bs_image_embedding,
            image_pe = image_pe,
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings = dense_embeddings,
            multimask_output = False
        )
        origin_size = sampled_batch['data_samples'][0].ori_shape
        logits_origin = F.interpolate(
            logits_256,
            origin_size,
            mode="bilinear",
            align_corners=False,
        )


        decoder_outputs = dict(
            logits_256 = logits_256,
            logits_origin = logits_origin,
            iou_pred = iou_pred,
            embeddings_256 = extra_outputs['upscaled_embedding'],
            decoder_inner_embeds = extra_outputs['decoder_inner_embeds'],
            inter_features = inter_features
        )
        
        return decoder_outputs


    def gene_img_embed(self, sampled_batch: dict):
        
        if self.use_embed:
            bs_image_embedding = torch.stack(sampled_batch['img_embed']).to(self.device)
        else:
            with torch.no_grad():
                # input_images.shape: [bs, 3, 1024, 1024]
                image_tensor = torch.stack(sampled_batch['inputs'])  # (bs, 3, 1024, 1024), 3 is bgr
                image_tensor_rgb = image_tensor[:, [2, 1, 0], :, :]
                input_images = self.preprocess(image_tensor_rgb).to(self.device)  
                # bs_image_embedding.shape: [bs, c=256, h=64, w=64]
                bs_image_embedding,inter_features = self.image_encoder(input_images, need_inter=True)
                return bs_image_embedding,inter_features

        return bs_image_embedding,None
    