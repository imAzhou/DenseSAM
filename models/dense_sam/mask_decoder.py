# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type

from models.sam.modeling import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        transformer_dim: int = 256,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        encoder_embed_dim: int = 1280,
        use_inner_feat = False,
        use_boundary_head = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.use_inner_feat = use_inner_feat
        self.num_mask_tokens = 1
        self.use_boundary_head = use_boundary_head

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # origin paramters
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # new paramters
        if use_inner_feat:
            embed_dim, out_chans = encoder_embed_dim, transformer_dim
            self.process_inter_feat = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, 
                          groups = embed_dim, padding='same'),
                nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim//2, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(embed_dim//2, out_chans, kernel_size=1)
            )
        if use_boundary_head:
            self.output_boundary_mlps = nn.ModuleList(
                [
                    MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                    for i in range(self.num_mask_tokens)
                ]
            )
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        inter_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          inter_feature (torch.Tensor): the inner embeddings from the image encoder

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        
        outputs = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings = sparse_prompt_embeddings,
            dense_prompt_embeddings = dense_prompt_embeddings,
            inter_feature = inter_feature
        )
        masks,iou_pred = outputs['logits_256'],outputs['iou_pred']

        # Select the correct mask or masks for output
        mask_slice = slice(0, None)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        outputs['logits_256'],outputs['iou_pred'] = masks,iou_pred
        
        return outputs

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        inter_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        predict masks for <image_embeddings, prompt> pairs.

        The pairs will be automatically generated in the following cases:
            1. one-vs-all: In single image with multiple prompts, image embedding will be broadcasted to n_prompts
            2. all-vs-one: In multiple images with single prompt, prompt will be broadcasted to batch_size
        In multiple images with multiple prompts, the inputs will be regarded as pairs and not be broadcasted.

        Arguments:
            image_embeddings (torch.Tensor): [B1 x embed_dim x embed_h x embed_w] image embeddings.
            image_pe (torch.Tensor): [1 x embed_dim x embed_h x embed_w] position encoding. 
            According to the code, it is a learnable image-independent parameter.
            sparse_prompt_embeddings (torch.Tensor): [B2 x N x embed_dim] sparse prompt embeddings 
            with token length N. 
            dense_prompt_embeddings (torch.Tensor): [B2 x embed_dim x embed_h] dense prompt embeddings
            inter_feature (torch.Tensor): the inner embeddings from the image encoder
        """
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        b1, b2 = image_embeddings.shape[0], sparse_prompt_embeddings.shape[0]
        if b1 > 1 and b2 == 1:
            tokens = torch.repeat_interleave(tokens, b1, dim=0)
            dense_prompt_embeddings = torch.repeat_interleave(dense_prompt_embeddings, b1, dim=0)
        elif b2 > 1 and b1 == 1:
            image_embeddings = torch.repeat_interleave(image_embeddings, b2, dim=0)
        elif b1 != b2:
            raise ValueError(f'The input embeddings pairs cannot be automatically generated! Get image_embeddings with shape {image_embeddings.shape} and sparse_embeddings with shape {sparse_prompt_embeddings.shape}')

        src = image_embeddings + dense_prompt_embeddings
        b, c, h, w = src.shape
        pos_src = torch.repeat_interleave(image_pe, b, dim=0)
        
        # Run the transformer
        inter_feat_c256 = None
        if self.use_inner_feat:
            inter_feat_c256 = self.process_inter_feat(inter_feature.permute(0, 3, 1, 2))
        
        hs, src = self.transformer(src, pos_src, tokens, inter_feat_c256)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        outputs = {
            'logits_256': masks,
            'iou_pred': iou_pred,
            'src': src,
            'upscaled_embedding': upscaled_embedding,
            'mask_tokens_out': mask_tokens_out,
        }

        if self.use_boundary_head:
            boundary_in_list: List[torch.Tensor] = []
            for i in range(self.num_mask_tokens):
                boundary_in_list.append(self.output_boundary_mlps[i](mask_tokens_out[:, i, :]))
            boundary_in = torch.stack(boundary_in_list, dim=1)
            b, c, h, w = upscaled_embedding.shape
            boundary_masks = (boundary_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
            outputs['logits_256'] = torch.cat((masks,boundary_masks),dim=1)

        return outputs



# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
