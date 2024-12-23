import torch
from torch import nn
from PromptUNet.PromptEncoder import *
import time
import random
import copy
import numpy as np

class ResidualCrossConnection(ResidualConnection):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
            super().__init__(d_model,dropout)

    def forward(self,x,y,sublayer,attn_mask = False):
        output = self.dropout(sublayer(x,y,attn_mask))
        return self.norm(x + output)

class ResidualCrossConnectionMap(ResidualConnection):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
            super().__init__(d_model,dropout)

    def forward(self,x,y,sublayer,attn_mask = False):
        output = sublayer(x,y,attn_mask)
        attn_map = output[1]
        output = self.dropout(output[0])
        return self.norm(x + output),attn_map

class ResidualPosConnection(ResidualConnection):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__(d_model, dropout)

    def forward(self, x, pos,add_pos, sublayer):
        output = self.dropout(sublayer(x))
        if add_pos:
            output += pos
        return self.norm(x + output)
class Embeddings(nn.Module):

    def __init__(self,d_model, size,device):
        super().__init__()
        self.d_model = d_model
        self.num_labels = size[0]*size[1]
        self.embedding = nn.Embedding(self.num_labels,d_model,device=device)

    def forward(self,x):
        return self.embedding(x)


class MultiHeadCrossAttentionLayer(nn.Module):
    def __init__(self, d_model=384, num_heads=4, dropout=0.1):
        super().__init__()

        # self.w_k = nn.Linear(d_model, d_model, bias=False)
        # self.w_q = nn.Linear(d_model, d_model, bias=False)
        # self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.attn_layer = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)



    def forward(self, q_tensors,kv_tensors,attn_mask = False):  # x_shape = B,L+1,d_model
        #
        # Q = self.w_q(q_tensors)
        # K = self.w_k(kv_tensors)
        # V = self.w_v(kv_tensors)

        if torch.is_tensor(attn_mask):
            attn_output = self.attn_layer(q_tensors,kv_tensors,kv_tensors,need_weights=False,attn_mask=attn_mask)
        else:
            attn_output = self.attn_layer(q_tensors,kv_tensors,kv_tensors, need_weights=False)


        return attn_output[0]


class CrossAttentionBlock(nn.Module):

    def __init__(self, d_model=384, num_heads=4, dropout=0.1):
        super().__init__()
        self.CrossAttention1 = MultiHeadCrossAttentionLayer(d_model,num_heads,dropout)
        self.FFN = FFN(d_model,dropout)
        self.res_connection1 = ResidualCrossConnection(d_model,dropout)
        self.res_connection2 = ResidualConnection(d_model,dropout)
        self.CrossAttention2 = MultiHeadCrossAttentionLayer(d_model,num_heads,dropout)
        self.FFN2 = FFN(d_model, dropout)
        self.res_connection3 = ResidualCrossConnection(d_model, dropout)
        self.res_connection4 = ResidualPosConnection(d_model, dropout)


    def forward(self,images,prompts_input,original_prompts,pos_encodings,add_pos=True,attn_mask=False,prompt_attn_mask=False):


        # images = self.res_connection3(images, prompts_input, self.CrossAttention2,attn_mask )
        # images = self.res_connection4(images, pos_encodings, add_pos, self.FFN2)
        #
        # images = self.res_connection4(images, pos_encodings, add_pos, self.FFN2)
        #
        # if torch.is_tensor(attn_mask):
        #     prompt_attn_mask = torch.transpose(attn_mask,1,2)
        # else:
        #     prompt_attn_mask = False
        #
        # prompts_input = self.res_connection1(prompts_input,images,self.CrossAttention1,prompt_attn_mask)
        # prompts_input = self.res_connection2(prompts_input,self.FFN)

        prompts_input = self.res_connection1(prompts_input, images, self.CrossAttention1,prompt_attn_mask)
        #print(f'prompt na: {torch.any(torch.isnan(prompts_input))}')
        prompts_input = self.res_connection2(prompts_input,self.FFN)
        prompts_output = prompts_input + original_prompts

        images = self.res_connection3(images, prompts_output, self.CrossAttention2,attn_mask)
        #print(f'image na: {torch.any(torch.isnan(images))}')
        images = self.res_connection4(images, pos_encodings, add_pos, self.FFN2)

        return images,prompts_output


class ImageEncoder(nn.Module):

    def __init__(self,device,embeddings,d_model=128,d_image_in=576, num_heads=8,num_blocks=6, dropout=0.1,):
        super().__init__()
        self.device = device
        self.d_model=d_model
        self.image_feature_embedding = nn.Linear(d_image_in,d_model)
        self.embedding_to_feature = nn.Linear(d_model,d_image_in)
        self.embeddings = embeddings
        self.cross_attns = nn.ModuleList([CrossAttentionBlock(d_model,num_heads,dropout) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(d_model)

    def pos_emb_grid(self,B, H, W):
        grid = torch.ones((H, W), device=self.device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / H
        x_embed = x_embed / W

        pe = self.embeddings._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        if self.embeddings.use_mlp:
            pe = self.embeddings.mlp(pe)
        pe = pe.permute(2, 0, 1)  # d_model x H x W

        pe = pe.unsqueeze(0).repeat(B, 1, 1, 1)  # shape Bxd_modelxHxW

        return pe

    def forward(self,images_input,prompts,original_prompts):
        B,D,H,W = images_input.shape

        pe = self.pos_emb_grid(B,H,W)



        images_input = torch.permute(images_input,(0,2,3,1)) # rearange to BxHxWxD for linear projection


        images_input = self.image_feature_embedding(images_input)
        #images_input = images_input.reshape(B,H,W,self.d_model)


        images_input = torch.permute(images_input,(0,3,1,2)) # now shape B,d_model,H,W

        #images = images_input + pe # += is an in place alteration and messes up backprop

        images_input = images_input.view(B,self.d_model,-1)

        images_input = images_input.permute(0,2,1)
        pe = pe.view(B,self.d_model,-1)
        pe = pe.permute(0,2,1)
        images = images_input + pe
        images = self.ln(images)  # trying layernorm before first input into cross attention
        add_pos = True
        for i in range(len(self.cross_attns)):
            if i == len(self.cross_attns)-1:
                add_pos = False
            images,prompts = self.cross_attns[i](images,prompts,original_prompts,pe,add_pos)


        images = self.embedding_to_feature(images)
        images = torch.reshape(images,(B,D,H,W))

        return images,prompts




class maskedImageEncoder(ImageEncoder):
    def __init__(self, device, embeddings, d_model=128, d_image_in=576, num_heads=8, num_blocks=6, dropout=0.1,
                 radius=10, box=False):
        super().__init__(device, embeddings, d_model, d_image_in, num_heads, num_blocks, dropout)
        self.num_heads = num_heads
        self.radius = radius
        self.box = box

    def check_distance(self, feat_index, point_list, r):
        # diffs = torch.abs(feat_index[1:-1] - point_list[feat_index[0],:,:])
        point_list_indexed = torch.index_select(point_list, 0, feat_index[0].unsqueeze(0))

        # Use torch.unsqueeze to add a singleton dimension to feat_index[1:-1] and point_list_indexed
        feat_index_unsqueezed = torch.unsqueeze(feat_index[1:-1], 0)
        point_list_unsqueezed = torch.unsqueeze(point_list_indexed, 0)

        # Subtract the tensors and compute the absolute value
        diffs = torch.abs(feat_index_unsqueezed - point_list_unsqueezed)

        dist = torch.max(diffs, dim=3)[0]

        dist_mask = torch.logical_not(torch.le(dist, r)).to(torch.float)

        dist_mask = dist_mask.repeat(self.num_heads, 1, 1)

        return dist_mask

    def forward(self, images_input, prompts, original_prompts, points):
        B, D, H, W = images_input.shape

        pe = self.pos_emb_grid(B, H, W)

        check_distance_v = torch.vmap(self.check_distance, in_dims=(0, None, None))

        points /= (512 / H)  # assuming input image is square, matches point to feature resolution ( .to(torch.int) will truncate decimal values later)
        points = points.to(torch.int)

        num_points = points.shape[1]
        L = num_points * (self.radius + 1) ** 2
        add_max = L - (self.radius // 2 + 1) ** 2

        images_input = torch.permute(images_input, (0, 2, 3, 1))  # rearange to BxHxWxD
        # images_input = self.image_feature_embedding(images_input)

        point_map = torch.zeros((B, H, W))

        for b in range(B):

            for point_idx in range(num_points):
                i, j = points[b, point_idx, 0].to(torch.int), points[b, point_idx, 1].to(torch.int)

                i_min = max(0, i - self.radius // 2)
                j_min = max(0, j - self.radius // 2)
                i_max = min(H, i + self.radius // 2)
                j_max = min(W, j + self.radius // 2)

                point_map[b, i_min:i_max + 1, j_min:j_max + 1].fill_(1)
            #  count = int(torch.sum(point_map[b, :, :]).item())

        point_counts = torch.sum(point_map, dim=(1, 2))

        point_map = point_map.bool().to(self.device).unsqueeze(3)
        pe = pe.permute(0, 2, 3, 1)

        cumsum = torch.cumsum(point_counts, dim=0)

        # Create a list to store the padded tensors
        padded = []
        padded_pe = []

        # Loop over each batch
        # move selected vectors to new tensor so we dont apply linear projection to cross attention dimension to all vectors needlessly
        masked_output = torch.masked_select(images_input, point_map)
        masked_pe = torch.masked_select(pe, point_map)

        padding_mask = torch.zeros((B, L))
        for i in range(B):
            padding_mask[i, point_counts[i].to(torch.int):].fill_(1)

        prompt_padding_mask = padding_mask.repeat(self.num_heads, 1).unsqueeze(2).to(self.device)  # now shape B*num_heads,L,1

        padding_mask = padding_mask.bool().to(self.device)

        for i in range(B):
            # Get the output tensor for the current batch, shape: (points[i], D)
            # Use the cumulative sum to index the transformed tensor
            if i == 0:
                output = masked_output[:cumsum[i].to(torch.int) * D]
                pe_output = masked_pe[:cumsum[i].to(torch.int) * self.d_model]
            else:
                output = masked_output[cumsum[i - 1].to(torch.int) * D:cumsum[i].to(torch.int) * D]
                pe_output = masked_pe[cumsum[i - 1].to(torch.int) * self.d_model:cumsum[i].to(torch.int) * self.d_model]

            output = torch.reshape(output, (point_counts[i].to(torch.int), D))
            pe_output = torch.reshape(pe_output, (point_counts[i].to(torch.int), self.d_model))
            # Create a padding tensor of zeros, shape: (padding[i], D)

            pad = torch.zeros(L - point_counts[i].to(torch.int), D).to(self.device)

            # Concatenate the output and padding tensors along the second dimension, shape: (L, D)
            concat = torch.cat([output, pad], dim=0)
            concat_pe = torch.cat([pe_output, pad[:, :self.d_model]], dim=0)

            # Append the concatenated tensor to the list
            padded.append(concat)
            padded_pe.append(concat_pe)

        # Stack the padded tensors along the first dimension, shape: (B, L, D)
        masked_output = torch.stack(padded, dim=0).to(self.device)
        masked_pe = torch.stack(padded_pe, dim=0).to(self.device)

        masked_output = self.image_feature_embedding(masked_output)

        masked_output = masked_output + masked_pe

        masked_output = self.ln(masked_output)  # trying layernorm before first input into cross attention

        selected_feats = torch.argwhere(point_map)

        unique_batches, f_counts = torch.unique(selected_feats[:, 0], return_counts=True)
        feat_counts = torch.cat([torch.arange(val) for val in f_counts])
        selected_feats[:,-1] = feat_counts

        check = check_distance_v(selected_feats, points, self.radius // 2)


        attention_mask = torch.zeros((self.num_heads * B, L, num_points)).to(self.device)

        batch_indices = (selected_feats[:, 0] * self.num_heads).unsqueeze(1).expand(-1, self.num_heads)
        head_indices = batch_indices + torch.arange(self.num_heads).unsqueeze(0).to(self.device)
        flat_head_indices = head_indices.view(-1)
        expanded_selected_feats = selected_feats[:, -1].unsqueeze(1).expand(-1, self.num_heads).reshape(-1)

        attention_mask[flat_head_indices.to(torch.long), expanded_selected_feats.to(torch.long), :] = check.view(-1,num_points)


        prompt_attention_mask = copy.deepcopy(attention_mask)
        prompt_attention_mask += prompt_padding_mask
        prompt_attention_mask = torch.transpose(prompt_attention_mask,1,2).to(self.device)



        attention_mask = attention_mask.bool().to(self.device)

        add_pos = True
        for i in range(len(self.cross_attns)):

            if i == len(
                    self.cross_attns) - 1:  # no need to add positional encoding after final cross attention layer -> positional encodings of next set of patches are different resolution
                add_pos = False

            masked_output, prompts = self.cross_attns[i](masked_output, prompts, original_prompts, masked_pe, add_pos,
                                                         attention_mask,prompt_attention_mask)

        masked_output = self.embedding_to_feature(masked_output)
        masked_output = torch.masked_select(masked_output, torch.logical_not(padding_mask).unsqueeze(2))
        images_input = images_input.masked_scatter(point_map, masked_output)

        images_input = images_input.permute(0, 3, 1, 2)
        return images_input, prompts



class MultiHeadCrossAttentionLayerMap(nn.Module):
    def __init__(self, d_model=384, num_heads=4, dropout=0.1):
        super().__init__()

        self.attn_layer = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)



    def forward(self, q_tensors,kv_tensors,attn_mask=False):  # x_shape = B,L+1,d_model

        if torch.is_tensor(attn_mask):
            attn_output = self.attn_layer(q_tensors,kv_tensors,kv_tensors,need_weights=True,average_attn_weights=True,attn_mask=attn_mask)
        else:
            attn_output = self.attn_layer(q_tensors,kv_tensors,kv_tensors, need_weights=True,average_attn_weights=True)
        attn_output = (attn_output[0],attn_output[1])#[:,0,:,:]) # select one of the heads
        return attn_output
class CrossAttentionBlockMap(CrossAttentionBlock):

    def __init__(self, d_model=384, num_heads=4, dropout=0.1):
        super().__init__(d_model,num_heads,dropout)
        self.CrossAttention1 = MultiHeadCrossAttentionLayerMap(d_model,num_heads,dropout)
        self.CrossAttention2 = MultiHeadCrossAttentionLayerMap(d_model,num_heads,dropout)
        self.res_connection1 = ResidualCrossConnectionMap(d_model,dropout)
        self.res_connection3 = ResidualCrossConnectionMap(d_model,dropout)
    def forward(self,images,prompts_input,original_prompts,pos_encodings,add_pos=True,attn_mask=False,prompt_attn_mask=False):

        prompts_input,prompt_to_image_attn = self.res_connection1(prompts_input, images, self.CrossAttention1,prompt_attn_mask)  # if using masked encoder need to hide empty image K-V pairs
        prompts_output = self.res_connection2(prompts_input, self.FFN)
        prompts_output = prompts_output + original_prompts
        #print(f'prompt post  leaked nan: {torch.isnan(prompts_output).any()}')
        #print(f'image pre leaked nan: {torch.isnan(images).any()}')
        images,image_to_prompt_attn = self.res_connection3(images, prompts_output, self.CrossAttention2, attn_mask)
       # print(f'image post leaked nan: {torch.isnan(images).any()}')
        images = self.res_connection4(images, pos_encodings, add_pos, self.FFN2)


        return images, prompts_output,prompt_to_image_attn,image_to_prompt_attn
class ImageEncoderMap(ImageEncoder):
    def __init__(self, device, embeddings, d_model=128, d_image_in=576, num_heads=8, num_blocks=6, dropout=0.1,):
        super().__init__(device,embeddings,d_model,d_image_in,num_heads,num_blocks,dropout)
        self.cross_attns = nn.ModuleList([CrossAttentionBlockMap(d_model, num_heads, dropout) for _ in range(num_blocks)])

    def forward(self,images_input,prompts,original_prompts,name,attn_maps=False):
        B,D,H,W = images_input.shape
        #print(f'b: {B} D {D} H {H} W {W}')

        pe = self.pos_emb_grid(B,H,W)

        #print(f'pe shape: {pe.shape}')

        images_input = torch.permute(images_input,(0,2,3,1)) # rearange to BxHxWxD so when we flatten vectors are sorted by batch

        images_input = self.image_feature_embedding(images_input)
        #images_input = images_input.reshape(B,H,W,self.d_model)

        images_input = torch.permute(images_input,(0,3,1,2)) # now shape B,d_model,H,W

        #images = images_input + pe # += is an in place alteration and messes up backprop

        images_input = images_input.view(B,self.d_model,-1)

        images_input = images_input.permute(0,2,1)
        pe = pe.view(B,self.d_model,-1)
        pe = pe.permute(0,2,1)
        images = images_input + pe

        images = self.ln(images)  # trying layernorm before first input into cross attention

        add_pos = True
        #print(f'input shape: {images.shape}')

        for i in range(len(self.cross_attns)):
            if i == len(self.cross_attns)-1:
                add_pos = False

            images,prompts,prompt_to_image_attn,image_to_prompt_attn = self.cross_attns[i](images,prompts,original_prompts,pe,add_pos)
            if attn_maps:
                t = int(time.time())
                path_p_i = f'/data_hd1/students/felix_cohen/attention_maps/{name}_prompt_to_image_layer_{i}_{t}'
                path_i_p = f'/data_hd1/students/felix_cohen/attention_maps/{name}_image_to_prompt_layer_{i}_{t}'
                np.save(path_p_i,prompt_to_image_attn.numpy())
                np.save(path_i_p,image_to_prompt_attn.numpy())
                attn_maps[f'{name} prompt to image layer {i}'].append(path_p_i)
                attn_maps[f'{name} image to prompt layer {i}'].append(path_i_p)
        images = self.embedding_to_feature(images)
        images = torch.reshape(images,(B,D,H,W))

        return images,prompts
