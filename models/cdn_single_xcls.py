import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from mmdet.core import bbox2roi
from util.box_ops import box_cxcywh_to_xyxy, union_box
import itertools

from models.word_vectors import obj_edge_vectors
from util.label_set import OBJ_CLASSES
import pdb


class CDN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers_hopd=3, num_dec_layers_interaction=3, 
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, dsgg_task='sgcls', args=None):
        super().__init__()

        self.dsgg_task = dsgg_task
        self.one_dec = args.one_dec
        self.use_roi = args.use_roi

        # Transformer structure
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_dec_layers_hopd, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        if not self.one_dec:
            interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            interaction_decoder_norm = nn.LayerNorm(d_model)
            self.interaction_decoder = TransformerDecoder(interaction_decoder_layer, num_dec_layers_interaction, interaction_decoder_norm,
                                                return_intermediate=return_intermediate_dec)

        # initialize decoder query with roi feature
        # if self.use_roi:
        #     layer_cfg = dict(type='RoIAlign', output_size=7, sampling_ratio=2)
        #     layer_type = layer_cfg.pop("type")
        #     layer_cls = getattr(ops, layer_type)
        #     self.roi_layer = layer_cls(spatial_scale=1 / 32, **layer_cfg)
        #     self.roi_query_proj = nn.Sequential(
        #         nn.Conv2d(d_model, d_model, kernel_size=3),
        #         nn.LeakyReLU(0.2, inplace=True),
        #         nn.Conv2d(d_model, d_model, kernel_size=3),
        #         nn.LeakyReLU(0.2, inplace=True),
        #         nn.Conv2d(d_model, d_model, kernel_size=3),
        #     )

        # intialize query embedding with object class embedding and query position with spatial information
        self.register_buffer('obj_classes_embed', obj_edge_vectors(OBJ_CLASSES, wv_type='glove.6B', wv_dir='./data/', wv_dim=200))
        self.query_spatial = MLP(12, d_model, d_model, 2)
        if self.dsgg_task == 'predcls':
            if self.use_roi:
                self.query_content = MLP(2 * d_model, d_model, d_model, 2)
            else:
                self.query_content = MLP(200, d_model, d_model, 2)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, targets):
        bs, c, h, w = src.shape
        # pdb.set_trace()
        # # >>>>> construct query embedding <<<<<
        # # | SGCLS: using ROI feature of union box
        # # | PredCLS: using ROI feature of union box and object class embedding 
        # union_boxes_list = []
        # for bid in range(bs):
        #     target = targets[bid]
        #     sub_boxes_xyxy = box_cxcywh_to_xyxy(target['sub_boxes'])
        #     obj_boxes_xyxy = box_cxcywh_to_xyxy(target['obj_boxes'])
        #     union_boxes_xyxy = union_box(sub_boxes_xyxy, obj_boxes_xyxy)
        #     imgs_whwh_shape = target['size'][[0, 1, 0, 1]]
        #     union_boxes_xyxy = union_boxes_xyxy * imgs_whwh_shape
        #     union_boxes_list.append(union_boxes_xyxy)
        # union_rois = bbox2roi(union_boxes_list)
        # union_roi_feats = self.roi_layer(src, union_rois)
        # roi_query_embed = self.roi_query_proj(union_roi_feats).squeeze()

        # target_num_list = [len(union_boxes_list[i]) for i in range(len(union_boxes_list))]
        # if 0 in target_num_list:
        #     pdb.set_trace()
        # roi_query_embed_list = torch.split(roi_query_embed, target_num_list, dim=0)
        # padded_roi_query_embed = torch.nn.utils.rnn.pad_sequence(roi_query_embed_list) # check size sequence
        # query_padding_mask = torch.ones((bs, len(padded_roi_query_embed)), device=memory.device, dtype=torch.bool)
        # for bid in range(bs):
        #     query_padding_mask[bid, :target_num_list[bid]] = False

        # >>>>> image encoder <<<<<
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask_for_roi = mask
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        if self.use_roi:
            memory_for_roi = memory.permute(1, 2, 0).unsqueeze(-1).view(bs, self.d_model, h, w).contiguous()

        # >>>>> construct query embedding <<<<<
        # | SGCLS: only initialize query position embedding by spatial information over exustive object pairs
        # | PredCLS: initialize query embedding with object class embedding and query position with spatial information
        valid_query_num_list = []
        for i in range(bs):
            if self.dsgg_task == 'predcls':
                num_pairs = len(targets[i]['obj_labels'])
            elif self.dsgg_task == 'sgcls':
                num_boxes = len(targets[i]['boxes'])
                num_pairs = num_boxes * (num_boxes - 1)
            else:
                raise NotImplementedError
            valid_query_num_list.append(num_pairs)
        max_len = max(valid_query_num_list)
        query_pos = torch.zeros((max_len, bs, 12), device=src.device)
        query_padding_mask = torch.ones((bs, max_len), device=src.device, dtype=torch.bool)
        if self.dsgg_task == 'predcls':
            if not self.use_roi:
                query_embed = torch.zeros((max_len, bs, 200), device=src.device)
            else:
                query_embed = torch.zeros((max_len, bs, 2 * self.d_model), device=src.device)
        else:
            query_embed = torch.zeros((max_len, bs, self.d_model), device=src.device)
        if self.dsgg_task == 'predcls':
            for bid in range(bs):
                target = targets[bid]
                if self.use_roi:
                    # obj_embedding = []
                    # sub_idx = torch.where(target['labels'] == 1)[0]
                    # obj_idx = torch.where(target['labels'] != 1)[0]
                    # sub_boxes_xyxy = box_cxcywh_to_xyxy(target['boxes'][sub_idx])
                    # obj_boxes_xyxy = box_cxcywh_to_xyxy(target['boxes'][obj_idx])
                    # so_boxes_xyxy = torch.cat([sub_boxes_xyxy, obj_boxes_xyxy], dim=0)
                    # imgs_whwh_shape = target['size'][[1, 0, 1, 0]]
                    # so_boxes_xyxy = so_boxes_xyxy * imgs_whwh_shape
                    # so_rois = bbox2roi([so_boxes_xyxy])
                    # m_h = torch.logical_not(mask_for_roi[bid, :, 0]).sum()
                    # m_w = torch.logical_not(mask_for_roi[bid, 0]).sum()
                    # cropped_memory = memory_for_roi[bid:bid+1][:, :, :m_h, :m_w] # remove masked part
                    # so_roi_feats = self.roi_layer(cropped_memory, so_rois) 
                    # so_roi_embed = self.roi_query_proj(so_roi_feats).squeeze()
                    # for oid in range(len(obj_idx)):
                    #     cat_embed = torch.cat([so_roi_embed[sub_idx[0]], so_roi_embed[obj_idx[oid]]], dim=-1)
                    #     obj_embedding.append(cat_embed)
                    # obj_embedding = torch.stack(obj_embedding, dim=0)
                    pdb.set_trace()
                else:
                    obj_embedding = self.obj_classes_embed[target['obj_labels']]
                spatial_pos = torch.cat([target['sub_boxes'], target['obj_boxes'], target['sub_boxes'][:, :2] - target['obj_boxes'][:, :2], \
                    torch.prod(target['sub_boxes'][:, 2:], dim=-1, keepdim=True), torch.prod(target['obj_boxes'][:, 2:], dim=-1, keepdim=True)], dim=-1)
                num_pairs = valid_query_num_list[bid]
                query_embed[:num_pairs, bid] = obj_embedding
                query_pos[:num_pairs, bid] = spatial_pos
                query_padding_mask[bid, :num_pairs] = False
            query_embed = self.query_content(query_embed)
        
        elif self.dsgg_task == 'sgcls':
            for bid in range(bs):
                target = targets[bid]
                num_boxes = len(target['boxes'])
                num_pairs = valid_query_num_list[bid]
                pair_indices = torch.tensor(list(itertools.permutations(range(num_boxes), 2)))
                sub_boxes = target['boxes'][pair_indices[:, 0]]
                obj_boxes = target['boxes'][pair_indices[:, 1]]
                spatial_pos = torch.cat([sub_boxes, obj_boxes, sub_boxes[:, :2] - obj_boxes[:, :2], \
                    torch.prod(sub_boxes[:, 2:], dim=-1, keepdim=True), torch.prod(obj_boxes[:, 2:], dim=-1, keepdim=True)], dim=-1)                
                query_pos[:num_pairs, bid] = spatial_pos
                query_padding_mask[bid, :num_pairs] = False

        query_pos = self.query_spatial(query_pos)


        # >>>>> instance decoder <<<<<
        # tgt = torch.zeros_like(padded_roi_query_embed)
        tgt = query_embed
        hopd_out = self.decoder(tgt, memory, tgt_key_padding_mask=query_padding_mask, memory_key_padding_mask=mask, 
                                pos=pos_embed, query_pos=query_pos)
        hopd_out = hopd_out.transpose(1, 2)

        if self.one_dec:
            return hopd_out, valid_query_num_list
        
        # # >>>>> relation decoder <<<<<
        interaction_query_embed = hopd_out[-1]
        interaction_query_embed = interaction_query_embed.permute(1, 0, 2)

        interaction_tgt = torch.zeros_like(interaction_query_embed)
        # interaction_tgt = query_embed
        interaction_decoder_out = self.interaction_decoder(interaction_tgt, memory, tgt_key_padding_mask=query_padding_mask, 
                                                           memory_key_padding_mask=mask, pos=pos_embed, query_pos=interaction_query_embed)
        interaction_decoder_out = interaction_decoder_out.transpose(1, 2)

        return hopd_out, interaction_decoder_out, valid_query_num_list #, memory.permute(1, 2, 0).view(bs, c, h, w)


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

def build_cdn(args):
    return CDN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers_hopd=args.dec_layers_hopd,
        num_dec_layers_interaction=args.dec_layers_interaction,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        dsgg_task=args.dsgg_task,
        args=args
    )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
