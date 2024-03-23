import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.position_encoding import PositionalEncoding3D, PositionalEncoding1D

import pdb


class CDN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers_hopd=3, num_dec_layers_interaction=3, 
                 num_dec_layers_temporal=3, num_ref_frames=3,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None, matcher=None):
        super().__init__()

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

        # Temporal Interaction Module
        self.num_dec_layers_temporal = num_dec_layers_temporal
        self.query_temporal_interaction = args.query_temporal_interaction
        self.temporal_feature_encoder = args.temporal_feature_encoder
        self.instance_temporal_interaction = args.instance_temporal_interaction
        self.relation_temporal_interaction = args.relation_temporal_interaction
        self.seq_sort = args.seq_sort
        self.one_temp = args.one_temp

        if self.temporal_feature_encoder:
            self.temporal_encoder = copy.deepcopy(decoder_layer)
        
        if self.seq_sort:
                self.spatial_temporal_pe = PositionalEncoding3D(d_model) # TODO: whether need 3d positional encoding
                if self.query_temporal_interaction:
                    self.temporal_pe = PositionalEncoding1D(2 * d_model)
                else:
                    self.temporal_pe = PositionalEncoding1D(d_model)
        if self.query_temporal_interaction:
            temporal_query_layer = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
            if num_dec_layers_temporal == 1:
                if self.one_temp:
                    self.temporal_query_layer = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
                else:
                    self.temporal_query_layer1 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
                    self.temporal_query_layer2 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
                    self.temporal_query_layer3 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
            else:
                self.temporal_query_decoder1 = QueryTransformerDecoder(temporal_query_layer, num_dec_layers_temporal)
                self.temporal_query_decoder2 = QueryTransformerDecoder(temporal_query_layer, num_dec_layers_temporal)
                self.temporal_query_decoder3 = QueryTransformerDecoder(temporal_query_layer, num_dec_layers_temporal)
            
            # self.ins_temporal_interaction_decoder = TransformerDecoder(decoder_layer, num_dec_layers_temporal, nn.LayerNorm(d_model))
            # self.rel_temporal_interaction_decoder = TransformerDecoder(decoder_layer, num_dec_layers_temporal, nn.LayerNorm(d_model))
        else:
            if self.instance_temporal_interaction:
                self.ins_temporal_query_layer1 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.ins_temporal_query_layer2 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.ins_temporal_query_layer3 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.ins_temporal_interaction_decoder = TransformerDecoder(decoder_layer, num_dec_layers_temporal, nn.LayerNorm(d_model))
            if self.relation_temporal_interaction:
                self.rel_temporal_query_layer1 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.rel_temporal_query_layer2 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.rel_temporal_query_layer3 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.rel_temporal_interaction_decoder = TransformerDecoder(decoder_layer, num_dec_layers_temporal, nn.LayerNorm(d_model))
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.num_ref_frames = num_ref_frames
        self.use_matched_query = args.use_matched_query

        # # for check something
        self.matcher = matcher
        # self.gt_hit_k1 = 0
        # self.gt_hit_k2 = 0
        # self.gt_hit_k3 = 0
        # self.gt_hit_k4 = 0
        # self.gt_hit_k5 = 0
        # self.gt_hit_k6 = 0
        # self.gt_total = 0
        # self.check_frames = 5000

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, class_embed_dict=None, targets=None, cur_idx=0):
        
        # Class Embedding MLP for coarse-to-fine query interaction
        obj_class_embed = class_embed_dict['obj_class_embed']
        attn_class_embed = class_embed_dict['attn_class_embed']
        spatial_class_embed = class_embed_dict['spatial_class_embed']
        contacting_class_embed = class_embed_dict['contacting_class_embed']
        sub_bbox_embed = class_embed_dict['sub_bbox_embed']
        obj_bbox_embed = class_embed_dict['obj_bbox_embed']

        n, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        st_pos_embed = pos_embed
        query_embed = query_embed.unsqueeze(1).repeat(1, n, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # bs = memory.shape[1] // (self.num_ref_frames + 1)
        # memory_list = torch.stack(torch.chunk(memory, bs, dim=1), dim=0)
        # pdb.set_trace()
        # cur_memory = memory_list[:, :, cur_idx:cur_idx+1]
        cur_memory = memory[:, cur_idx:cur_idx+1]
        ref_ids = torch.as_tensor((list(range(cur_idx)) + list(range(cur_idx+1, self.num_ref_frames+1))), dtype=torch.long, device=memory.device)
        ref_memory = memory[:, ref_ids].transpose(0, 1).reshape(-1, 1, self.d_model)
        # ref_memory = memory_list[:, :, ref_ids]

        # Pair-wise Instance Decoder
        hopd_out = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        hopd_out = hopd_out.transpose(1, 2)
        # pdb.set_trace()

        # # Frame Feature Temporal Interaction
        # if self.temporal_feature_encoder:
        #     ref_mask = mask[ref_ids].flatten()[None, :]
        #     cur_mask = mask[cur_idx][None, :]
        #     if self.seq_sort:
        #         st_pos_embed = self.spatial_temporal_pe((n, h, w, self.d_model)).to(pos_embed.device)
        #         st_pos_embed = st_pos_embed.permute(0, 3, 1, 2).flatten(2).permute(2, 0, 1)
        #     cur_st_pos_embed = st_pos_embed[:, cur_idx:cur_idx+1]
        #     ref_st_pos_embed = st_pos_embed[:, ref_ids]
        #     ref_st_pos_embed_list = torch.chunk(ref_st_pos_embed, self.num_ref_frames, dim=1)
        #     ref_st_pos_embed = torch.cat(ref_st_pos_embed_list, 0)
        #     cur_memory = self.temporal_encoder(cur_memory, ref_memory, tgt_key_padding_mask=cur_mask, memory_key_padding_mask=ref_mask, \
        #                                        pos=ref_st_pos_embed, query_pos=cur_st_pos_embed) # 3d pe

        last_ins_hs = hopd_out[-1]
        # last_ins_hs = torch.stack(torch.chunk(last_ins_hs, bs, dim=0), dim=0)
        cur_ins_hs = last_ins_hs[cur_idx:cur_idx+1]
        ref_ins_hs = last_ins_hs[ref_ids]
        ref_ins_hs_list = torch.chunk(ref_ins_hs, self.num_ref_frames, dim=0)
        ref_ins_hs = torch.cat(ref_ins_hs_list, 1)
        
        cur_hs_tpe, ref_hs_tpe = None, None
        if self.seq_sort:
            if self.query_temporal_interaction:
                hs_tpe = self.temporal_pe((n, 2 * self.d_model)).to(last_ins_hs.device)
            else:
                hs_tpe = self.temporal_pe((n, self.d_model)).to(last_ins_hs.device)
            hs_tpe = hs_tpe[:, None].repeat(1, last_ins_hs.shape[1], 1)
            cur_hs_tpe = hs_tpe[cur_idx:cur_idx+1]
            ref_hs_tpe = hs_tpe[ref_ids]
            ref_hs_tpe_list = torch.chunk(ref_hs_tpe, self.num_ref_frames, dim=0)
            ref_hs_tpe = torch.cat(ref_hs_tpe_list, 1)

        # Pair-wise Instance Temporal Interaction
        # final_ins_hs = cur_ins_hs
        # if self.instance_temporal_interaction:
        #     ref_obj_prob = obj_class_embed(ref_ins_hs).softmax(-1)[..., :-1].max(-1)[0]

        #     _, topk_indexes = torch.topk(ref_obj_prob, 80 * self.num_ref_frames, dim=1)
        #     ref_ins_hs_input1 = torch.gather(ref_ins_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_ins_hs.shape[-1]))
        #     ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
        #     cur_ins_hs_tmp = self.ins_temporal_query_layer1(cur_ins_hs, ref_ins_hs_input1, cur_hs_tpe, ref_hs_tpe_topk)  # 1d pe for query interaction

        #     _, topk_indexes = torch.topk(ref_obj_prob, 50 * self.num_ref_frames, dim=1)
        #     ref_ins_hs_input2 = torch.gather(ref_ins_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_ins_hs.shape[-1]))
        #     ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
        #     cur_ins_hs_tmp = self.ins_temporal_query_layer2(cur_ins_hs_tmp, ref_ins_hs_input2, cur_hs_tpe, ref_hs_tpe_topk)

        #     _, topk_indexes = torch.topk(ref_obj_prob, 30 * self.num_ref_frames, dim=1)
        #     ref_ins_hs_input3 = torch.gather(ref_ins_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_ins_hs.shape[-1]))
        #     ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
        #     cur_ins_hs_tmp = self.ins_temporal_query_layer3(cur_ins_hs_tmp, ref_ins_hs_input3, cur_hs_tpe, ref_hs_tpe_topk)

        #     cur_ins_hs_tmp = cur_ins_hs_tmp.permute(1, 0, 2)
        #     ins_tgt = torch.zeros_like(cur_ins_hs_tmp)
        #     cur_ins_hs_tmp = self.ins_temporal_interaction_decoder(ins_tgt, cur_memory, memory_key_padding_mask=mask[cur_idx:cur_idx+1],
        #                             pos=pos_embed[:, cur_idx:cur_idx+1], query_pos=cur_ins_hs_tmp) # 2d pe in spatial encoder
        #     cur_ins_hs_tmp = cur_ins_hs_tmp.transpose(0, 1)

        #     final_ins_hs = cur_ins_hs_tmp
        #     #### cur_ins_hs = cur_ins_hs_tmp

        # Relation Decoder
        interaction_query_embed = torch.cat([hopd_out[-1][:cur_idx], cur_ins_hs, hopd_out[-1][cur_idx+1:]])
        interaction_query_embed = interaction_query_embed.permute(1, 0, 2)
        interaction_tgt = torch.zeros_like(interaction_query_embed)
        interaction_decoder_out = self.interaction_decoder(interaction_tgt, memory, memory_key_padding_mask=mask,
                                        pos=pos_embed, query_pos=interaction_query_embed) # 2d pe as no temopral interaction in spatial decoder
        interaction_decoder_out = interaction_decoder_out.transpose(1, 2)
        last_rel_hs = interaction_decoder_out[-1]
        cur_rel_hs = last_rel_hs[cur_idx:cur_idx+1]
        ref_rel_hs = last_rel_hs[ref_ids]
        ref_rel_hs_list = torch.chunk(ref_rel_hs, self.num_ref_frames, dim=0)
        ref_rel_hs = torch.cat(ref_rel_hs_list, 1)

        # # Relation Temporal Interaction
        # if self.relation_temporal_interaction:
        #     if not self.instance_temporal_interaction:
        #         ref_obj_prob = obj_class_embed(ref_ins_hs).softmax(-1)[..., :-1].max(-1)[0]
        #     ref_attn_prob = attn_class_embed(ref_rel_hs).softmax(-1)[..., :-1].max(-1)[0]
        #     ref_spatial_prob = spatial_class_embed(ref_rel_hs).sigmoid().max(-1)[0]
        #     ref_contacting_prob = contacting_class_embed(ref_rel_hs).sigmoid().max(-1)[0]

        #     overall_probs = ref_obj_prob * ref_attn_prob * ref_spatial_prob * ref_contacting_prob

        #     _, topk_indexes = torch.topk(overall_probs, 80 * self.num_ref_frames, dim=1) # 80
        #     ref_rel_hs_input1 = torch.gather(ref_rel_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_rel_hs.shape[-1]))
        #     ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
        #     cur_rel_hs = self.rel_temporal_query_layer1(cur_rel_hs, ref_rel_hs_input1, cur_hs_tpe, ref_hs_tpe_topk)

        #     _, topk_indexes = torch.topk(overall_probs, 50 * self.num_ref_frames, dim=1)
        #     ref_rel_hs_input2 = torch.gather(ref_rel_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_rel_hs.shape[-1]))
        #     ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
        #     cur_rel_hs = self.rel_temporal_query_layer2(cur_rel_hs, ref_rel_hs_input2, cur_hs_tpe, ref_hs_tpe_topk)

        #     _, topk_indexes = torch.topk(overall_probs, 30 * self.num_ref_frames, dim=1)
        #     ref_rel_hs_input3 = torch.gather(ref_rel_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_rel_hs.shape[-1]))
        #     ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
        #     cur_rel_hs = self.rel_temporal_query_layer3(cur_rel_hs, ref_rel_hs_input3, cur_hs_tpe, ref_hs_tpe_topk)

        #     ## Code for stats: how many queries matched with ground truth are selected as reference query
        #     ## matching between gt and prediction at reference frames 
        #     # ref_obj_logits = obj_class_embed(ref_ins_hs)
        #     # ref_sub_bboxes = sub_bbox_embed(ref_ins_hs).sigmoid()
        #     # ref_obj_bboxes = obj_bbox_embed(ref_ins_hs).sigmoid()
        #     # ref_attn_logits = attn_class_embed(ref_rel_hs)
        #     # ref_spatial_logits = spatial_class_embed(ref_rel_hs)
        #     # ref_contacting_logits = contacting_class_embed(ref_rel_hs)
        #     # matched_ref_index_list = []
        #     # for i in range(self.num_ref_frames):
        #     #     outputs_ref_i = {'pred_obj_logits': ref_obj_logits[:, i*100:(i+1)*100, :], 'pred_sub_boxes': ref_sub_bboxes[:, i*100:(i+1)*100, :],\
        #     #                      'pred_obj_boxes': ref_obj_bboxes[:, i*100:(i+1)*100, :], 'pred_attn_logits': ref_attn_logits[:, i*100:(i+1)*100, :],\
        #     #                      'pred_spatial_logits': ref_spatial_logits[:, i*100:(i+1)*100, :], 'pred_contacting_logits': ref_contacting_logits[:, i*100:(i+1)*100, :]}
        #     #     targets_ref_i = [targets[0][i+1]]
        #     #     indices_ref_i = self.matcher(outputs_ref_i, targets_ref_i)
        #     #     self.gt_total += len(indices_ref_i[0][1])
        #     #     for idx in indices_ref_i[0][0]:
        #     #         matched_ref_index_list.append(idx.item() + i*100)
        #     # _, topk1_indexes = torch.topk(overall_probs, 5 * self.num_ref_frames, dim=1)
        #     # _, topk2_indexes = torch.topk(overall_probs, 10 * self.num_ref_frames, dim=1)
        #     # _, topk3_indexes = torch.topk(overall_probs, 20 * self.num_ref_frames, dim=1)
        #     # _, topk4_indexes = torch.topk(overall_probs, 30 * self.num_ref_frames, dim=1)
        #     # _, topk5_indexes = torch.topk(overall_probs, 50 * self.num_ref_frames, dim=1)
        #     # _, topk6_indexes = torch.topk(overall_probs, 80 * self.num_ref_frames, dim=1)

        #     # for idx in matched_ref_index_list:
        #     #     if idx in topk1_indexes:
        #     #         self.gt_hit_k1 += 1
        #     #     if idx in topk2_indexes:
        #     #         self.gt_hit_k2 += 1
        #     #     if idx in topk3_indexes:
        #     #         self.gt_hit_k3 += 1
        #     #     if idx in topk4_indexes:
        #     #         self.gt_hit_k4 += 1
        #     #     if idx in topk5_indexes:
        #     #         self.gt_hit_k5 += 1
        #     #     if idx in topk6_indexes:
        #     #         self.gt_hit_k6 += 1
        #     # self.check_frames -= 1
        #     # if self.check_frames == 0:
        #     #     print('--------------------------')
        #     #     print('gt_hit_10: ', self.gt_hit_k1, '; recall: ', self.gt_hit_k1 / self.gt_total)
        #     #     print('gt_hit_20: ', self.gt_hit_k2, '; recall: ', self.gt_hit_k2 / self.gt_total)
        #     #     print('gt_hit_30: ', self.gt_hit_k3, '; recall: ', self.gt_hit_k3 / self.gt_total)
        #     #     print('gt_hit_50: ', self.gt_hit_k4, '; recall: ', self.gt_hit_k4 / self.gt_total)
        #     #     print('gt_hit_80: ', self.gt_hit_k5, '; recall: ', self.gt_hit_k5 / self.gt_total)
        #     #     print('gt_total: ', self.gt_total)
        #     #     print('--------------------------')
        #     #     pdb.set_trace()
        #     ##  sel_indexes = torch.as_tensor(matched_ref_index_list, dtype=torch.int64)[None, :]
        #     ##  sel_indexes = sel_indexes.to(ref_rel_hs.device)
        #     ##  ref_rel_hs_input = torch.gather(ref_rel_hs, 1, sel_indexes.unsqueeze(-1).repeat(1, 1, cur_rel_hs.shape[-1]))
        #     ##  cur_rel_hs = self.rel_temporal_query_layer1(cur_rel_hs, ref_rel_hs_input)
            
        #     cur_rel_hs = cur_rel_hs.permute(1, 0, 2)
        #     rel_tgt = torch.zeros_like(cur_rel_hs)
        #     cur_rel_hs = self.rel_temporal_interaction_decoder(rel_tgt, cur_memory, memory_key_padding_mask=mask[cur_idx:cur_idx+1],
        #                             pos=pos_embed[:, cur_idx:cur_idx+1], query_pos=cur_rel_hs)
        #     cur_rel_hs = cur_rel_hs.transpose(0, 1)
        # final_rel_hs = cur_rel_hs

        if self.query_temporal_interaction:
            ref_obj_prob = obj_class_embed(ref_ins_hs).softmax(-1)[..., :-1].max(-1)[0]
            ref_attn_prob = attn_class_embed(ref_rel_hs).softmax(-1)[..., :-1].max(-1)[0]
            ref_spatial_prob = spatial_class_embed(ref_rel_hs).sigmoid().max(-1)[0]
            ref_contacting_prob = contacting_class_embed(ref_rel_hs).sigmoid().max(-1)[0]

            cur_concat_hs = torch.cat([cur_ins_hs, cur_rel_hs], dim=-1)
            ref_concat_hs = torch.cat([ref_ins_hs, ref_rel_hs], dim=-1)

            overall_probs = ref_obj_prob * ref_attn_prob * ref_spatial_prob * ref_contacting_prob

            if self.one_temp:
                if self.use_matched_query:
                    ref_obj_logits = obj_class_embed(ref_ins_hs)
                    ref_sub_bboxes = sub_bbox_embed(ref_ins_hs).sigmoid()
                    ref_obj_bboxes = obj_bbox_embed(ref_ins_hs).sigmoid()
                    ref_attn_logits = attn_class_embed(ref_rel_hs)
                    ref_spatial_logits = spatial_class_embed(ref_rel_hs)
                    ref_contacting_logits = contacting_class_embed(ref_rel_hs)

                    matched_ref_index_list = []
                    for i in range(self.num_ref_frames):
                        outputs_ref_i = {'pred_obj_logits': ref_obj_logits[:, i*100:(i+1)*100, :], 'pred_sub_boxes': ref_sub_bboxes[:, i*100:(i+1)*100, :],\
                                        'pred_obj_boxes': ref_obj_bboxes[:, i*100:(i+1)*100, :], 'pred_attn_logits': ref_attn_logits[:, i*100:(i+1)*100, :],\
                                        'pred_spatial_logits': ref_spatial_logits[:, i*100:(i+1)*100, :], 'pred_contacting_logits': ref_contacting_logits[:, i*100:(i+1)*100, :]}
                        targets_ref_i = [targets[i+1]]
                        indices_ref_i = self.matcher(outputs_ref_i, targets_ref_i)
                        for idx in indices_ref_i[0][0]:
                            matched_ref_index_list.append(idx.item() + i*100)
                    sel_indexes = torch.as_tensor(matched_ref_index_list, dtype=torch.int64)[None, :]
                    sel_indexes = sel_indexes.to(ref_rel_hs.device)
                if not self.use_matched_query:
                    sel_num = 10 * self.num_ref_frames
                    _, sel_indexes = torch.topk(overall_probs, sel_num, dim=1)
                ref_concat_hs_input = torch.gather(ref_concat_hs, 1, sel_indexes.unsqueeze(-1).repeat(1, 1, ref_concat_hs.shape[-1]))
                ref_hs_tpe_topk = ref_hs_tpe[:, sel_indexes[0]] if self.seq_sort else None
                if self.num_dec_layers_temporal == 1:
                    cur_concat_hs = self.temporal_query_layer(cur_concat_hs, ref_concat_hs_input, cur_hs_tpe, ref_hs_tpe_topk)
                else:
                    cur_concat_hs = self.temporal_query_decoder(cur_concat_hs, ref_concat_hs_input, cur_hs_tpe, ref_hs_tpe_topk)

                cur_concat_hs = cur_concat_hs.permute(1, 0, 2)
                cur_ins_hs, cur_rel_hs = torch.split(cur_concat_hs, self.d_model, -1)
            
            else:
                _, topk_indexes = torch.topk(overall_probs, 80 * self.num_ref_frames, dim=1) # 80
                ref_concat_hs_input1 = torch.gather(ref_concat_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_concat_hs.shape[-1]))
                ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
                if self.num_dec_layers_temporal == 1:
                    cur_concat_hs = self.temporal_query_layer1(cur_concat_hs, ref_concat_hs_input1, cur_hs_tpe, ref_hs_tpe_topk)
                else:
                    cur_concat_hs = self.temporal_query_decoder1(cur_concat_hs, ref_concat_hs_input1, cur_hs_tpe, ref_hs_tpe_topk)

                _, topk_indexes = torch.topk(overall_probs, 50 * self.num_ref_frames, dim=1)
                ref_concat_hs_input2 = torch.gather(ref_concat_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_concat_hs.shape[-1]))
                ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
                if self.num_dec_layers_temporal == 1:
                    cur_concat_hs = self.temporal_query_layer2(cur_concat_hs, ref_concat_hs_input2, cur_hs_tpe, ref_hs_tpe_topk)
                else:
                    cur_concat_hs = self.temporal_query_decoder2(cur_concat_hs, ref_concat_hs_input2, cur_hs_tpe, ref_hs_tpe_topk)

                _, topk_indexes = torch.topk(overall_probs, 30 * self.num_ref_frames, dim=1)
                ref_concat_hs_input3 = torch.gather(ref_concat_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_concat_hs.shape[-1]))
                ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
                if self.num_dec_layers_temporal == 1:
                    cur_concat_hs = self.temporal_query_layer3(cur_concat_hs, ref_concat_hs_input3, cur_hs_tpe, ref_hs_tpe_topk)
                else:
                    cur_concat_hs = self.temporal_query_decoder3(cur_concat_hs, ref_concat_hs_input3, cur_hs_tpe, ref_hs_tpe_topk)

                cur_concat_hs = cur_concat_hs.permute(1, 0, 2)
                cur_ins_hs, cur_rel_hs = torch.split(cur_concat_hs, self.d_model, -1)

                # tgt = torch.zeros_like(cur_ins_hs)
                # cur_ins_hs = self.ins_temporal_interaction_decoder(tgt, cur_memory, memory_key_padding_mask=mask[cur_idx:cur_idx+1],
                #                         pos=pos_embed[:, cur_idx:cur_idx+1], query_pos=cur_ins_hs)
                # cur_rel_hs = self.rel_temporal_interaction_decoder(tgt, cur_memory, memory_key_padding_mask=mask[cur_idx:cur_idx+1],
                #                         pos=pos_embed[:, cur_idx:cur_idx+1], query_pos=cur_rel_hs)
            cur_ins_hs = cur_ins_hs.transpose(0, 1)
            cur_rel_hs = cur_rel_hs.transpose(0, 1)
            final_ins_hs = cur_ins_hs
            final_rel_hs = cur_rel_hs

        return final_ins_hs, final_rel_hs


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
    
    def forward(self, query, ref_query, query_pos=None, ref_query_pos=None):
        # self.attention
        q = k = self.with_pos_embed(query, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), query.transpose(0, 1))[0].transpose(0, 1)
        tgt = query + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention 
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos).transpose(0, 1), 
            self.with_pos_embed(ref_query, ref_query_pos).transpose(0, 1),
            ref_query.transpose(0,1)
        )[0].transpose(0,1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

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
