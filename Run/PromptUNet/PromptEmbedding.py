import torch
from torch import nn
from typing import Any, Optional, Tuple, Type
import numpy as np
class PositionEmbeddingRandom(nn.Module):

    """
    Positional encoding scheme using random spatial frequencies. - Adapted from SAM paper by facebook
    """

    """
    Use_mlp boolean determines whether we use the modulating positional encoding scheme created by Li et al. - found this makes little to no difference 
    Scale determines the variance of the gaussian used to create the matrix used in pos encoding scheme - see the paper: 'Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains'
    num_pos_feats determines dimension of positional encoding scheme
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None, use_mlp: bool = False,new_dim=384) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0

        self.use_mlp = use_mlp
        #self.sigma = torch.nn.Parameter(torch.tensor(1.0))
        if self.use_mlp:
            pos_feat_out = num_pos_feats*2
            num_pos_feats = 64
            self.l1 = torch.nn.Linear(num_pos_feats * 2, num_pos_feats*2)
            self.relu = torch.nn.ReLU()
            self.l_out = torch.nn.Linear(num_pos_feats*2, pos_feat_out)
            self.mlp = torch.nn.Sequential(self.l1, self.relu, self.l_out)

        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )
        self.B = torch.nn.Parameter(scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape

        coords = 2 * coords - 1

        if not self.use_mlp:
            coords = coords @  self.positional_encoding_gaussian_matrix
        else:
            coords=coords @ self.B
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords = coords.flip(2) # turn ij coordinats to xy
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]

        if not self.use_mlp:
            return self._pe_encoding(coords.to(torch.float))  # B x N x C
        else:
            return self.mlp(self._pe_encoding(coords.to(torch.float)))




class PromptEmbedder(nn.Module):
    def __init__(
        self,pe_layer,
        embed_dim: int,
        input_image_size: Tuple[int, int],device,num_point_embeddings=3
    ) -> None:

        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size

        self.pe_layer = pe_layer # for every b_i vector, a cos(b_i^Tx) and sin(b_i^Tx) output is created therefore half embed_dim

        self.num_point_embeddings: int = num_point_embeddings  # cup disc and background classes
        point_embeddings = [nn.Embedding(1, embed_dim,device=device) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim) # for padding class


    def forward(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # point_embedding[
        #     labels == -1] = 0.0  # set padded point embeddings to 0 (have been put thru point_embedding func)
        # point_embedding[
        #     labels == -1] += self.not_a_point_embed.weight  # set padded point embeddings to not a point signifier


        point_embedding[labels.squeeze(-1) == 0] += self.point_embeddings[0].weight
        point_embedding[labels.squeeze(-1) == 0] -= self.point_embeddings[1].weight
        point_embedding[labels.squeeze(-1) == 0] -= self.point_embeddings[2].weight

        point_embedding[labels.squeeze(-1) == 1] += self.point_embeddings[1].weight
        point_embedding[labels.squeeze(-1) == 1] -= self.point_embeddings[0].weight
        point_embedding[labels.squeeze(-1) == 1] -= self.point_embeddings[2].weight

        point_embedding[labels.squeeze(-1) == 2] += self.point_embeddings[2].weight
        point_embedding[labels.squeeze(-1) == 2] -= self.point_embeddings[0].weight
        point_embedding[labels.squeeze(-1) == 2] -= self.point_embeddings[1].weight

        return point_embedding


class BoxPromptEmbedder(PromptEmbedder):
    def __init__(
        self,pe_layer,
        embed_dim: int,
        input_image_size: Tuple[int, int],device,num_point_embeddings=7,
    ) -> None:
        super().__init__(pe_layer,embed_dim,input_image_size,device,num_point_embeddings)


    def forward(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # point_embedding[
        #     labels == -1] = 0.0  # set padded point embeddings to 0 (have been put thru point_embedding func)
        # point_embedding[
        #     labels == -1] += self.not_a_point_embed.weight  # set padded point embeddings to not a point signifier


        point_embedding[labels.squeeze(-1) == 0] += self.point_embeddings[0].weight
        point_embedding[labels.squeeze(-1) == 1] += self.point_embeddings[1].weight
        point_embedding[labels.squeeze(-1) == 2] += self.point_embeddings[2].weight
        point_embedding[labels.squeeze(-1) == 3] += self.point_embeddings[3].weight
        # point_embedding[labels.squeeze(-1) == 4] += self.point_embeddings[4].weight
        # point_embedding[labels.squeeze(-1) == 5] += self.point_embeddings[5].weight
        # point_embedding[labels.squeeze(-1) == 6] += self.point_embeddings[6].weight

        return point_embedding





