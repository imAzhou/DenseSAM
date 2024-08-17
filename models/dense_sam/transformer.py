# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor, nn
import math
from typing import Tuple, Type
from models.sam.modeling import MLPBlock,Attention


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        sm_depth: int = 1,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    sm_depth = sm_depth,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, 
            attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        inter_feat_c256: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding
        if inter_feat_c256 is not None:
            inter_feat_c256 = inter_feat_c256.flatten(2).permute(0, 2, 1)
        
        # Apply transformer blocks and final layernorm
        for i,layer in enumerate(self.layers):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
            
            if inter_feat_c256 is not None:
                keys += inter_feat_c256
        
        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)

        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        sm_depth: int = 1
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.sm_depth = sm_depth
        if sm_depth > 0:
            self.semantic_module = SemanticModule(embedding_dim, sm_depth)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        
        queries = self.norm1(queries)

        if self.sm_depth > 0:
            semantic_keys = self.semantic_module(keys)
            keys = keys + semantic_keys

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)

        # azhou add
        # if self.sm_depth > 0:
        #     semantic_keys = self.semantic_module(keys)
        #     keys = keys + semantic_keys + attn_out
        # else:
        #     keys = keys + attn_out
        
        keys = self.norm4(keys)

        # if self.sm_depth > 0:
        #     semantic_keys = self.semantic_module(keys)
        #     keys = keys + semantic_keys

        return queries, keys

class SemanticModule(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        sm_depth: int,
    ) -> None:
        super().__init__()
               
        self.local_conv = nn.ModuleList()
        self.sm_depth = sm_depth
        for i in range(self.sm_depth):
            block = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim*2, kernel_size=1),
                nn.BatchNorm2d(embedding_dim*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(embedding_dim*2, embedding_dim*2, kernel_size=3, groups=embedding_dim*2, padding='same'),
                nn.BatchNorm2d(embedding_dim*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(embedding_dim*2, embedding_dim, kernel_size=1),
                nn.BatchNorm2d(embedding_dim),
            )
            self.local_conv.append(block)
        
    def forward(self, tokens: Tensor) -> Tensor:
        
        b,l,c = tokens.shape
        patch_size = int(math.sqrt(l))
        local_conv_f = tokens.transpose(-1,-2).view(b, c, patch_size, patch_size)
        for block in self.local_conv:
            local_conv_f = block(local_conv_f)
        semantic_tokens = local_conv_f.flatten(2).permute(0, 2, 1)

        return semantic_tokens
  