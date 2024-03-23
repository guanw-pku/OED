import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.position_encoding import PositionalEncoding3D, PositionalEncoding1D
import itertools
from math import ceil

from models.word_vectors import obj_edge_vectors
from util.label_set import OBJ_CLASSES
from torch.nn.utils.rnn import pad_sequence
import pdb


class CDN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers_hopd=3, num_dec_layers_interaction=3, 
                 num_dec_layers_temporal=3, num_ref_frames=3,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None, matcher=None):
        super().__init__()

        self.dsgg_task = args.dsgg_task
        self.seq_sort = args.seq_sort
        self.no_update_pair = args.no_update_pair

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_dec_layers_hopd, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        interaction_decoder_norm = nn.LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(interaction_decoder_layer, num_dec_layers_interaction, interaction_decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        # intialize query embedding with object class embedding and query position with spatial information
        self.register_buffer('obj_classes_embed', obj_edge_vectors(OBJ_CLASSES, wv_type='glove.6B', wv_dir='./data/', wv_dim=200))
        self.query_spatial = MLP(12, d_model, d_model, 2)
        if self.dsgg_task == 'predcls':
            self.query_content = MLP(200, d_model, d_model, 2)

        # Temporal Interaction Module
        self.query_temporal_interaction = args.query_temporal_interaction
        if self.query_temporal_interaction:
            if self.dsgg_task == 'sgcls': 
                self.temporal_query_layer1 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
                self.temporal_query_layer2 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
                self.temporal_query_layer3 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
            else:
                # self.temporal_query_layer = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
                
                self.temporal_query_layer1 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
                self.temporal_query_layer2 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
                self.temporal_query_layer3 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)

                # self.temporal_query_layer = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
        
        self._reset_parameters()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_ref_frames = num_ref_frames

        if self.seq_sort:
            if self.query_temporal_interaction:
                self.temporal_pe = PositionalEncoding1D(2 * d_model)
            else:
                self.temporal_pe = PositionalEncoding1D(d_model)

        # for check something
        self.matcher = matcher

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, class_embed_dict=None, targets=None, cur_idx=0, learnable_query_embed=None):
        learnable_query_embed=None
        n_frames, c, h, w = src.shape
        # >>>>> construct query embedding <<<<<
        # | SGCLS: SGCLS: only initialize query position embedding by spatial information over exustive object pairs
        # | PredCLS: initialize query embedding with object class embedding and query position with spatial information
        valid_query_num_list = []
        for i in range(n_frames):
            if self.dsgg_task == 'predcls':
                num_pairs = len(targets[i]['obj_labels'])
            elif self.dsgg_task == 'sgcls':
                num_boxes = len(targets[i]['boxes'])
                num_pairs = num_boxes * (num_boxes - 1)
            else:
                raise NotImplementedError
            valid_query_num_list.append(num_pairs)
        ref_ids = torch.as_tensor((list(range(cur_idx)) + list(range(cur_idx+1, n_frames))), dtype=torch.long, device=src.device)
        ref_valid_query_num_list = [valid_query_num_list[i] for i in ref_ids]
        max_len = max(valid_query_num_list)
        query_pos = torch.zeros((max_len, n_frames, 12), device=src.device)
        query_padding_mask = torch.ones((n_frames, max_len), device=src.device, dtype=torch.bool)
        
        if self.dsgg_task == 'predcls':
            query_embed = torch.zeros((max_len, n_frames, 200), device=src.device)
            for tid in range(n_frames):
                target = targets[tid]
                obj_embedding = self.obj_classes_embed[target['obj_labels']]
                spatial_pos = torch.cat([target['sub_boxes'], target['obj_boxes'], target['sub_boxes'][:, :2] - target['obj_boxes'][:, :2], \
                    torch.prod(target['sub_boxes'][:, 2:], dim=-1, keepdim=True), torch.prod(target['obj_boxes'][:, 2:], dim=-1, keepdim=True)], dim=-1)
                num_pairs = valid_query_num_list[tid]
                query_embed[:num_pairs, tid] = obj_embedding
                query_pos[:num_pairs, tid] = spatial_pos
                query_padding_mask[tid, :num_pairs] = False
            query_embed = self.query_content(query_embed)
        elif self.dsgg_task == 'sgcls':
            query_embed = torch.zeros((max_len, n_frames, self.d_model), device=src.device)
            for tid in range(n_frames):
                target = targets[tid]
                num_boxes = len(target['boxes'])
                num_pairs = valid_query_num_list[tid]
                pair_indices = torch.tensor(list(itertools.permutations(range(num_boxes), 2)))
                sub_boxes = target['boxes'][pair_indices[:, 0]]
                obj_boxes = target['boxes'][pair_indices[:, 1]]
                spatial_pos = torch.cat([sub_boxes, obj_boxes, sub_boxes[:, :2] - obj_boxes[:, :2], \
                    torch.prod(sub_boxes[:, 2:], dim=-1, keepdim=True), torch.prod(obj_boxes[:, 2:], dim=-1, keepdim=True)], dim=-1)                
                query_pos[:num_pairs, tid] = spatial_pos
                query_padding_mask[tid, :num_pairs] = False
        query_pos = self.query_spatial(query_pos)

        # >>>>> image encoder <<<<<
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # >>>>> instance decoder <<<<<
        tgt = query_embed
        # if learnable_query_embed is not None:
        #     padded_query_pos = torch.zeros((learnable_query_embed.weight.shape[0], n_frames, learnable_query_embed.weight.shape[1]), device=src.device)
        #     padded_query_embed = torch.zeros((learnable_query_embed.weight.shape[0], n_frames, learnable_query_embed.weight.shape[1]), device=src.device)
        #     padded_padding_mask = torch.ones((n_frames, learnable_query_embed.weight.shape[0]), device=src.device, dtype=torch.bool)

        #     padded_query_pos[:max_len, :] = query_pos
        #     padded_query_embed[:max_len, :] = query_embed
        #     padded_padding_mask[:, :max_len] = query_padding_mask
            
        #     padded_query_pos[valid_query_num_list[cur_idx]:, cur_idx] = learnable_query_embed.weight[valid_query_num_list[cur_idx]:]
        #     padded_padding_mask[cur_idx] = False

        #     tgt = padded_query_embed
        #     query_padding_mask = padded_padding_mask
        #     query_pos = padded_query_pos
            
        hopd_out = self.decoder(tgt, memory, tgt_key_padding_mask=query_padding_mask, 
                            memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_pos)
        hopd_out = hopd_out.transpose(1, 2)

        # >>>>> relation decoder <<<<<
        interaction_query_embed = hopd_out[-1]
        interaction_query_embed = interaction_query_embed.permute(1, 0, 2)
        interaction_tgt = torch.zeros_like(interaction_query_embed)
        interaction_decoder_out = self.interaction_decoder(interaction_tgt, memory, tgt_key_padding_mask=query_padding_mask,
                                            memory_key_padding_mask=mask, pos=pos_embed, query_pos=interaction_query_embed)
        interaction_decoder_out = interaction_decoder_out.transpose(1, 2)

        # >>>>> time positional embedding <<<<<
        cur_hs_tpe, ref_hs_tpe = None, None
        if self.seq_sort:
            assert self.dsgg_task == 'predcls'

            if self.query_temporal_interaction:
                hs_tpe = self.temporal_pe((n_frames, 2 * self.d_model)).to(hopd_out.device)
            else:
                hs_tpe = self.temporal_pe((n_frames, self.d_model)).to(hopd_out.device)
            hs_tpe = hs_tpe[:, None].repeat(1, max_len, 1)
            if learnable_query_embed is None:
                cur_hs_tpe = hs_tpe[cur_idx:cur_idx+1, :valid_query_num_list[cur_idx]]
            else:
                cur_hs_tpe = hs_tpe[cur_idx:cur_idx+1, 0:1].repeat(1, learnable_query_embed.weight.shape[0], 1)
            ref_hs_tpe = hs_tpe[ref_ids]
            ref_hs_tpe = torch.cat([ref_hs_tpe[i:i+1, :num] for i, num in enumerate(ref_valid_query_num_list)], dim=1)

            # ref_hs_tpe_list = torch.chunk(ref_hs_tpe, self.num_ref_frames, dim=0)
            # ref_hs_tpe = torch.cat(ref_hs_tpe_list, 1)
        
        if self.query_temporal_interaction:
            last_ins_hs = hopd_out[-1]
            if learnable_query_embed is None:
                cur_ins_hs = last_ins_hs[cur_idx:cur_idx+1, :valid_query_num_list[cur_idx]] #[1, 100, 256]
            else:
                cur_ins_hs = last_ins_hs[cur_idx:cur_idx+1]
            ref_ins_hs = last_ins_hs[ref_ids]
            ref_ins_hs = torch.cat([ref_ins_hs[i:i+1, :num] for i, num in enumerate(ref_valid_query_num_list)], dim=1)

            last_rel_hs = interaction_decoder_out[-1]
            if learnable_query_embed is None:
                cur_rel_hs = last_rel_hs[cur_idx:cur_idx+1, :valid_query_num_list[cur_idx]]
            else:
                cur_rel_hs = last_rel_hs[cur_idx:cur_idx+1]
            ref_rel_hs = last_rel_hs[ref_ids]
            ref_rel_hs = torch.cat([ref_rel_hs[i:i+1, :num] for i, num in enumerate(ref_valid_query_num_list)], dim=1)
            
            if self.dsgg_task == 'sgcls':
                sub_class_embed = class_embed_dict['sub_class_embed']
                obj_class_embed = class_embed_dict['obj_class_embed']
                attn_class_embed = class_embed_dict['attn_class_embed']
                spatial_class_embed = class_embed_dict['spatial_class_embed']
                contacting_class_embed = class_embed_dict['contacting_class_embed']
                
                ref_sub_prob = sub_class_embed(ref_ins_hs).softmax(-1)[..., :-1].max(-1)[0]
                ref_obj_prob = obj_class_embed(ref_ins_hs).softmax(-1)[..., :-1].max(-1)[0]
                ref_attn_prob = attn_class_embed(ref_rel_hs).softmax(-1)[..., :-1].max(-1)[0]
                ref_spatial_prob = spatial_class_embed(ref_rel_hs).sigmoid().max(-1)[0]
                ref_contacting_prob = contacting_class_embed(ref_rel_hs).sigmoid().max(-1)[0]

                cur_concat_hs = torch.cat([cur_ins_hs, cur_rel_hs], dim=-1)
                ref_concat_hs = torch.cat([ref_ins_hs, ref_rel_hs], dim=-1)

                overall_probs = ref_sub_prob * ref_obj_prob * ref_attn_prob * ref_spatial_prob * ref_contacting_prob
                total_valid_ref = sum(valid_query_num_list[1:])
                _, topk_indexes = torch.topk(overall_probs, ceil(0.8 * total_valid_ref), dim=1)
                ref_concat_hs_input1 = torch.gather(ref_concat_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_concat_hs.shape[-1]))
                cur_concat_hs = self.temporal_query_layer1(cur_concat_hs, ref_concat_hs_input1)

                _, topk_indexes = torch.topk(overall_probs, ceil(0.5 * total_valid_ref), dim=1)
                ref_concat_hs_input2 = torch.gather(ref_concat_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_concat_hs.shape[-1]))
                cur_concat_hs = self.temporal_query_layer2(cur_concat_hs, ref_concat_hs_input2)

                _, topk_indexes = torch.topk(overall_probs, ceil(0.3 * total_valid_ref), dim=1)
                ref_concat_hs_input3 = torch.gather(ref_concat_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_concat_hs.shape[-1]))
                cur_concat_hs = self.temporal_query_layer3(cur_concat_hs, ref_concat_hs_input3)

                cur_concat_hs = cur_concat_hs.permute(1, 0, 2)
                cur_ins_hs, cur_rel_hs = torch.split(cur_concat_hs, self.d_model, -1)
                cur_ins_hs = cur_ins_hs.transpose(0, 1)
                cur_rel_hs = cur_rel_hs.transpose(0, 1)
            else:
                # tgt_mask, [cur_l, cur_l]
                # memory_mask, [cur_l, ref_l]
                # tgt_key_padding_mask, [1, cur_l]
                # memory_key_padding_mask [1, ref_l]
                
                # padded_labels = torch.cat([F.pad(target['obj_labels'], (0, max_len-valid_query_num_list[i]), 'constant', -1) for i,target in enumerate(targets)])
                # cur_labels = padded_labels[max_len*cur_idx:max_len*(cur_idx+1)]
                # ref_labels = torch.cat([padded_labels[:max_len*cur_idx], padded_labels[max_len*(cur_idx+1):]])
                # tgt_key_padding_mask = (cur_labels == -1)[None, :]
                # memory_key_padding_mask = (ref_labels == -1)[None, :]
                # memory_mask = (cur_labels[:, None] == ref_labels[None, :]) & (cur_labels[:, None] != -1)
                cur_labels = targets[cur_idx]['obj_labels']
                ref_labels = torch.cat([targets[idx]['obj_labels'] for idx in range(len(targets)) if idx != cur_idx])
                memory_mask = (cur_labels[:, None] != ref_labels[None, :])
                memory_mask[(memory_mask == True).all(dim=1)] = False

                cur_concat_hs = torch.cat([cur_ins_hs, cur_rel_hs], dim=-1)
                ref_concat_hs = torch.cat([ref_ins_hs, ref_rel_hs], dim=-1)
                cur_concat_hs = self.temporal_query_layer1(cur_concat_hs, ref_concat_hs, cur_hs_tpe, ref_hs_tpe,
                                                          memory_mask=memory_mask,)
                                                          # tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask
                                                          # memory_key_padding_mask=)
                cur_concat_hs = self.temporal_query_layer2(cur_concat_hs, ref_concat_hs, cur_hs_tpe, ref_hs_tpe,
                                                          memory_mask=memory_mask,)
                cur_concat_hs = self.temporal_query_layer3(cur_concat_hs, ref_concat_hs, cur_hs_tpe, ref_hs_tpe,
                                                          memory_mask=memory_mask,)

                cur_concat_hs = cur_concat_hs.permute(1, 0, 2)
                cur_ins_hs, cur_rel_hs = torch.split(cur_concat_hs, self.d_model, -1)
                cur_ins_hs = cur_ins_hs.transpose(0, 1)
                cur_rel_hs = cur_rel_hs.transpose(0, 1)
                if learnable_query_embed is not None:
                    cur_ins_hs = cur_ins_hs[:, :valid_query_num_list[cur_idx]]
                    cur_rel_hs = cur_rel_hs[:, :valid_query_num_list[cur_idx]]
                # cur_rel_hs = self.temporal_query_layer(cur_rel_hs, ref_rel_hs)

            final_ins_hs = cur_ins_hs
            final_rel_hs = cur_rel_hs

        if self.no_update_pair:
            return final_rel_hs, valid_query_num_list
        return final_ins_hs, final_rel_hs, valid_query_num_list


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class QueryTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                query_pos: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, query_pos=query_pos, ref_query_pos=pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TemporalQueryEncoderLayer(nn.Module):
    def __init__(self, d_model = 256, d_ffn = 1024, dropout=0.1, activation="relu", n_heads = 8):
        super().__init__()

        # self attention 
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # cross attention 
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn 
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model) 

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward(self, query, ref_query, query_pos=None, ref_query_pos=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        # self.attention
        q = k = self.with_pos_embed(query, query_pos)
        # pdb.set_trace()
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), 
                              query.transpose(0, 1), attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0].transpose(0, 1)
        tgt = query + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention 
        # if tgt.shape[1] != memory_mask.shape[0] or ref_query.shape[1] != memory_mask.shape[1]:
            # pdb.set_trace()
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos).transpose(0, 1), 
            self.with_pos_embed(ref_query, ref_query_pos).transpose(0, 1),
            ref_query.transpose(0,1), attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0].transpose(0,1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_cdn(args, matcher):
    return CDN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers_hopd=args.dec_layers_hopd,
        num_dec_layers_interaction=args.dec_layers_interaction,
        num_dec_layers_temporal=args.dec_layers_temporal,
        num_ref_frames=args.num_ref_frames,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        args=args,
        matcher=matcher
    )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
