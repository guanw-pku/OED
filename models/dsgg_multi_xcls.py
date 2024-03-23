from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc_multi import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import numpy as np
from queue import Queue
import math
import itertools
import copy
import pdb

from .backbone import build_backbone
from .matcher import build_matcher
from .cdn_multi_xcls import build_cdn


class CDNDSGG(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_queries, aux_loss=False, args=None):
        super().__init__()
        # task determines whether object pair detection results are given
        self.dsgg_task = args.dsgg_task
        self.no_update_pair = args.no_update_pair
        self.aux_learnable_query = args.aux_learnable_query

        # model parameters
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim) # no need for learnable query
        if self.dsgg_task == 'sgcls':
            self.sub_class_embed = nn.Linear(hidden_dim, 2)
            self.temporal_sub_class_embed = copy.deepcopy(self.sub_class_embed)
        
        if not self.no_update_pair:
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
            self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.attn_class_embed = nn.Linear(hidden_dim, args.num_attn_classes + 1) # add 1 for background
        self.spatial_class_embed = nn.Linear(hidden_dim, args.num_spatial_classes)
        self.contacting_class_embed = nn.Linear(hidden_dim, args.num_contacting_classes)
        
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

        # hyperparameters
        self.aux_loss = aux_loss
        self.use_matching = args.use_matching
        self.dec_layers_hopd = args.dec_layers_hopd
        self.dec_layers_interaction = args.dec_layers_interaction
        self.query_temporal_interaction = args.query_temporal_interaction
        assert self.query_temporal_interaction
        
        # temopral emebed head
        if not self.no_update_pair:
            self.temporal_obj_class_embed = copy.deepcopy(self.obj_class_embed)
            self.temporal_sub_bbox_embed = copy.deepcopy(self.sub_bbox_embed)
            self.temporal_obj_bbox_embed = copy.deepcopy(self.obj_bbox_embed)
        self.temporal_attn_class_embed = copy.deepcopy(self.attn_class_embed)
        self.temporal_spatial_class_embed = copy.deepcopy(self.spatial_class_embed)
        self.temporal_contacting_class_embed = copy.deepcopy(self.contacting_class_embed)


    def forward(self, samples: NestedTensor, targets=None, cur_idx=0):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        bs, c, h, w = samples.tensors.shape
        embed_dict = None
        if self.dsgg_task == 'sgcls':
            embed_dict = {'sub_class_embed': self.sub_class_embed, 'obj_class_embed': self.obj_class_embed, 
                          'attn_class_embed': self.attn_class_embed, 'spatial_class_embed': self.spatial_class_embed, 
                          'contacting_class_embed': self.contacting_class_embed, 'sub_bbox_embed': self.sub_bbox_embed, 
                          'obj_bbox_embed': self.obj_bbox_embed}
        if not self.no_update_pair:
            hopd_out, interaction_decoder_out, valid_num_list = self.transformer(self.input_proj(src), mask, pos[-1], embed_dict, targets, cur_idx, self.query_embed)
        else:
            interaction_decoder_out, valid_num_list = self.transformer(self.input_proj(src), mask, pos[-1], embed_dict, targets, cur_idx)

        out = {}
        if self.dsgg_task == 'sgcls':
            outputs_sub_class = self.temporal_sub_class_embed(hopd_out)
            out['pred_sub_logits'] = outputs_sub_class
        if not self.no_update_pair:
            outputs_obj_class = self.temporal_obj_class_embed(hopd_out)
            outputs_sub_coord = self.temporal_sub_bbox_embed(hopd_out).sigmoid()
            outputs_obj_coord = self.temporal_obj_bbox_embed(hopd_out).sigmoid()
            out.update({'pred_obj_logits': outputs_obj_class, 'pred_sub_boxes': outputs_sub_coord, 'pred_obj_boxes': outputs_obj_coord})
        outputs_attn_class = self.attn_class_embed(interaction_decoder_out)
        outputs_spatial_class = self.spatial_class_embed(interaction_decoder_out)
        outputs_contacting_class = self.contacting_class_embed(interaction_decoder_out)
        out.update({'pred_attn_logits': outputs_attn_class, 'pred_spatial_logits': outputs_spatial_class, 
                    'pred_contacting_logits': outputs_contacting_class, 'valid_num_list': valid_num_list})

        # if self.aux_loss:                      
        #     out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_sub_coord, outputs_obj_coord,
        #                                             outputs_attn_class, outputs_spatial_class,
        #                                             outputs_contacting_class)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_sub_coord, outputs_obj_coord, outputs_attn_class, outputs_spatial_class, outputs_contacting_class):
        min_dec_layers_num = min(self.dec_layers_hopd, self.dec_layers_interaction)
        return [{'pred_obj_logits': a, 'pred_sub_boxes': b, 'pred_obj_boxes': c, 'pred_attn_logits': d, \
                    'pred_spatial_logits': e, 'pred_contacting_logits': f}
                for a, b, c, d, e, f in zip(outputs_obj_class[-min_dec_layers_num : -1], outputs_sub_coord[-min_dec_layers_num : -1], \
                                            outputs_obj_coord[-min_dec_layers_num : -1], outputs_attn_class[-min_dec_layers_num : -1], \
                                            outputs_spatial_class[-min_dec_layers_num : -1], outputs_contacting_class[-min_dec_layers_num : -1])]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionDSGG(nn.Module):

    def __init__(self, num_obj_classes, num_queries, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.dsgg_task = args.dsgg_task

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_attn_classes = args.num_attn_classes
        self.num_spatial_classes = args.num_spatial_classes
        self.num_contacting_classes = args.num_contacting_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        if self.dsgg_task == 'sgcls':
            sub_empty_weight = torch.ones(2)
            sub_empty_weight[-1] = self.eos_coef
            self.register_buffer('sub_empty_weight', sub_empty_weight)

        self.alpha = args.alpha

        self.obj_nums_init = [0, 0, 13042, 8362, 10797, 15346, 8387, 6291, 24164, 11934, 16723, 25298, 
                              15515, 11849, 3023, 11343, 15610, 29641, 3192, 11065, 1488, 3371, 6172, 
                              9003, 15089, 3409, 7067, 3405, 10625, 8606, 5987, 10397, 30046, 3021, 
                              11019, 2545, 3651]
        self.attn_nums_init = [144546, 193213, 38724]
        self.spat_nums_init = [4643, 58176, 254476, 46368, 69810, 12921]
        self.cont_nums_init = [4008, 4076, 4377, 3214, 314, 156897, 11506, 3395, 105067, 8743, 40545, 7606, 52165, 86, 6761, 772, 1102]

        self.obj_reweight = args.obj_reweight
        self.rel_reweight = args.rel_reweight
        self.use_static_weights = args.use_static_weights
        
        Maxsize = args.queue_size

        if self.obj_reweight:
            self.q_obj = Queue(maxsize=Maxsize)
            self.p_obj = args.p_obj
            self.obj_weights_init = self.cal_weights(self.obj_nums_init, p=self.p_obj)

        if self.rel_reweight:
            self.p_rel = args.p_rel
            self.q_attn = Queue(maxsize=Maxsize)
            self.q_spat = Queue(maxsize=Maxsize)
            self.q_cont = Queue(maxsize=Maxsize)
            self.attn_weights_init = self.cal_weights(self.attn_nums_init, p=self.p_rel)
            self.spat_weights_init = self.cal_weights(self.spat_nums_init, p=self.p_rel)
            self.cont_weights_init = self.cal_weights(self.cont_nums_init, p=self.p_rel)

    def cal_weights(self, label_nums, p=0.5):
        num_fgs = len(label_nums[:-1])
        weight = [0] * (num_fgs + 1)
        num_all = sum(label_nums[:-1])

        for index in range(num_fgs):
            if label_nums[index] == 0: continue
            weight[index] = np.power(num_all/label_nums[index], p)

        weight = np.array(weight)
        weight = weight / np.mean(weight[weight>0])

        weight[-1] = np.power(num_all/label_nums[-1], p) if label_nums[-1] != 0 else 0

        weight = torch.FloatTensor(weight).cuda()
        return weight


    def loss_sub_labels(self, outputs, targets, indices, num_interactions, log=True):   
        assert 'pred_sub_logits' in outputs
        src_logits = outputs['pred_sub_logits']
        idx = self._get_src_permutation_idx(indices)
        valid_num_list = outputs['valid_num_list']
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes_s = torch.zeros_like(target_classes_o)
        sub_weights = self.sub_empty_weight

        if self.dsgg_task == 'predcls':
            loss_sub_ce = F.cross_entropy(src_logits[idx], target_classes_s, sub_weights)
        else:
            indexes_cumsum = [0]
            indexes_cumsum.extend(valid_num_list)
            indexes_cumsum = torch.cumsum(torch.tensor(indexes_cumsum), dim=0)
            valid_idx = indexes_cumsum[idx[0]] + idx[1]
            valid_src_logits = torch.cat([src_logits[i:i+1, :valid_num_list[i]] for i in range(len(targets))], dim=1)
            target_classes = torch.ones(valid_src_logits.shape[:2], dtype=torch.int64, device=valid_src_logits.device)
            target_classes[0, valid_idx] = target_classes_s

            loss_sub_ce = F.cross_entropy(valid_src_logits.transpose(1, 2), target_classes, sub_weights)
        losses = {'loss_sub_ce': loss_sub_ce}
        if log:
            losses['sub_class_error'] = 100 - accuracy(src_logits[idx], target_classes_s)[0]
        return losses
    
    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        obj_weights = self.empty_weight
        # if not self.obj_reweight:
        # elif self.use_static_weights:
        #     obj_weights = self.obj_nums_init
        # else:


        if self.dsgg_task == 'predcls':
            loss_obj_ce = F.cross_entropy(src_logits[idx], target_classes_o, obj_weights)
        else:
            indexes_cumsum = [0]
            valid_num_list = outputs['valid_num_list']
            indexes_cumsum.extend(valid_num_list)
            indexes_cumsum = torch.cumsum(torch.tensor(indexes_cumsum), dim=0)
            valid_idx = indexes_cumsum[idx[0]] + idx[1]
            valid_src_logits = torch.cat([src_logits[i:i+1, :valid_num_list[i]] for i in range(len(targets))], dim=1)
            target_classes = torch.full(valid_src_logits.shape[:2], self.num_obj_classes,
                                        dtype=torch.int64, device=valid_src_logits.device)
            target_classes[0, valid_idx] = target_classes_o
            loss_obj_ce = F.cross_entropy(valid_src_logits.transpose(1, 2), target_classes, obj_weights)
        losses = {'loss_obj_ce': loss_obj_ce}
        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_relation_labels(self, outputs, targets, indices, num_interactions):
        # loss = focal_loss(inputs, targets, gamma=self.args.action_focal_loss_gamma, alpha=self.args.action_focal_loss_alpha, prior_verb_label_mask=prior_verb_label_mask)
        num_attn_rel, num_spatial_rel, num_contacting_rel = 3, 6, 17
        attn_logits = outputs['pred_attn_logits'].reshape(-1, num_attn_rel + 1)
        spatial_logits = outputs['pred_spatial_logits'].reshape(-1, num_spatial_rel)
        contacting_logits = outputs['pred_contacting_logits'].reshape(-1, num_contacting_rel)
        attn_probs = attn_logits.softmax(dim=-1)
        spatial_probs = spatial_logits.sigmoid()
        contacting_probs = contacting_logits.sigmoid()

        idx = self._get_src_permutation_idx(indices)
        idx = (idx[0].to(attn_logits.device), idx[1].to(attn_logits.device))
        target_attn_classes_o = torch.cat([t['attn_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_spatial_classes_o = torch.cat([t['spatial_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_contacting_classes_o = torch.cat([t['contacting_labels'][J] for t, (_, J) in zip(targets, indices)])

        sel_idx = idx[0] * outputs['pred_attn_logits'].shape[1] + idx[1]
        attn_logits = attn_logits[sel_idx]
        spatial_probs = spatial_probs[sel_idx]
        contacting_probs = contacting_probs[sel_idx]
        target_attn_classes = target_attn_classes_o
        target_spatial_classes = target_spatial_classes_o
        target_contacting_classes = target_contacting_classes_o

        target_attn_labels = torch.where(target_attn_classes)[1]
        loss_attn_ce = F.cross_entropy(attn_logits, target_attn_labels)
        loss_spatial_ce = self._neg_loss(spatial_probs, target_spatial_classes, alpha=self.alpha)
        loss_contacting_ce = self._neg_loss(contacting_probs, target_contacting_classes, alpha=self.alpha)
        
        losses = {'loss_attn_ce': loss_attn_ce, 'loss_spatial_ce': loss_spatial_ce, 'loss_contacting_ce': loss_contacting_ce}

        return losses


    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'sub_labels': self.loss_sub_labels,
            'obj_labels': self.loss_obj_labels,
            'relation_labels': self.loss_relation_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        valid_num_list = outputs['valid_num_list']
        cur_idx = targets[0]['cur_idx']
        targets = [targets[cur_idx]]
        indices = []
        if self.dsgg_task == 'predcls':
            for tid in range(len(targets)):
                indices.append((torch.arange(len(targets[tid]['obj_labels'])), torch.arange(len(targets[tid]['obj_labels']))))
        else:
            for tid in range(len(targets)):
                valid_num = valid_num_list[tid]
                outputs_without_aux_i = {k: v[tid:tid+1, :valid_num] for k, v in outputs.items() if k != 'aux_outputs' and k != 'valid_num_list'}
                indices_i = self.matcher(outputs_without_aux_i, [targets[tid]])[0]
                indices.append(indices_i)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessDSGG(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id
        self.use_matching = args.use_matching
        self.dsgg_task = args.dsgg_task
        self.no_update_pair = args.no_update_pair

    @torch.no_grad()
    def forward(self, outputs, targets, cur_idx=0):
        target_sizes = torch.stack([t['size'] for t in targets], dim=0)
        num_valid_query = outputs['valid_num_list'][cur_idx]

        if not self.no_update_pair:
            out_obj_logits = outputs['pred_obj_logits']
            out_sub_boxes = outputs['pred_sub_boxes']
            out_obj_boxes = outputs['pred_obj_boxes']
            
            obj_prob = F.softmax(out_obj_logits, -1)
            obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        out_attn_logits = outputs['pred_attn_logits']
        out_spatial_logits = outputs['pred_spatial_logits']
        out_contacting_logits = outputs['pred_contacting_logits']

        assert len(out_attn_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        
        attn_probs = out_attn_logits[..., :-1].softmax(-1)
        spatial_probs = out_spatial_logits.sigmoid()
        contacting_probs = out_contacting_logits.sigmoid()

        if self.dsgg_task == 'sgcls':
            sub_obj_logits = outputs['pred_sub_logits']
            sub_prob = F.softmax(sub_obj_logits, -1)
            sub_scores, sub_labels = sub_prob[..., :-1].max(-1)

        results = []
        assert len(target_sizes) == 1
        for index in range(len(target_sizes)):
            frame_pred = {}
            # num_valid_query = valid_num_list[index]
            if not self.no_update_pair:
                frame_sub_boxes = out_sub_boxes[index][:num_valid_query]
                frame_obj_boxes = out_obj_boxes[index][:num_valid_query]
                frame_pred['pred_boxes'] = torch.cat([frame_sub_boxes, frame_obj_boxes], dim=0)
            frame_pred['pair_idx'] = torch.cat([torch.arange(num_valid_query)[:, None], \
                                            torch.arange(num_valid_query, 2*num_valid_query)[:, None]], dim=1).cpu().numpy()
            frame_pred['attention_distribution'] = attn_probs[index][:num_valid_query].cpu().numpy()
            frame_pred['spatial_distribution'] = spatial_probs[index][:num_valid_query].cpu().numpy()
            frame_pred['contacting_distribution'] = contacting_probs[index][:num_valid_query].cpu().numpy()
            if self.dsgg_task == 'predcls':
                if not self.no_update_pair:
                    frame_pred['pred_scores'] = torch.cat([torch.ones(num_valid_query), obj_scores[index][:num_valid_query].cpu()]).numpy()
                    frame_pred['pred_labels'] = torch.cat([torch.ones(num_valid_query), obj_labels[index][:num_valid_query].cpu()]).numpy()
                frame_pred['target_scores'] = torch.cat([torch.ones(num_valid_query), torch.ones(num_valid_query)]).cpu().numpy()
                frame_pred['target_labels'] = torch.cat([torch.ones(num_valid_query), targets[index]['obj_labels']]).cpu().numpy()
                if len(targets[index]['obj_labels']) != num_valid_query:
                    pdb.set_trace()
                frame_pred['target_boxes'] = torch.cat([targets[index]['sub_boxes'], targets[index]['obj_boxes']], dim=0).cpu().numpy()
            else:
                frame_pred['pred_scores'] = torch.cat([sub_scores[index][:num_valid_query], obj_scores[index][:num_valid_query].cpu()]).numpy()
                frame_pred['pred_labels'] = torch.cat([sub_labels[index][:num_valid_query], obj_labels[index][:num_valid_query].cpu()]).numpy()
                num_boxes = len(targets[index]['boxes'])
                pair_indices = torch.tensor(list(itertools.permutations(range(num_boxes), 2)))
                target_sub_boxes = targets[index]['boxes'][pair_indices[:, 0]]
                target_obj_boxes = targets[index]['boxes'][pair_indices[:, 1]]
                frame_pred['target_boxes'] = torch.cat([target_sub_boxes, target_obj_boxes], dim=0).cpu().numpy()

            results.append(frame_pred)

        return results


def build(args):

    if 'ag' in args.dataset_file:
        num_classes = 36 + 1

    device = torch.device(args.device)

    backbone = build_backbone(args)

    matcher = build_matcher(args)
    cdn = build_cdn(args, matcher)

    model = CDNDSGG(
        backbone,
        cdn,
        num_obj_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_attn_ce'] = args.rel_loss_coef
    weight_dict['loss_spatial_ce'] = args.rel_loss_coef
    weight_dict['loss_contacting_ce'] = args.rel_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.dsgg_task == 'sgcls':
        weight_dict['loss_sub_ce'] = args.obj_loss_coef

    if args.aux_loss:
        min_dec_layers_num = min(args.dec_layers_hopd, args.dec_layers_interaction)
        aux_weight_dict = {}
        for i in range(min_dec_layers_num - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['obj_labels', 'relation_labels', 'sub_obj_boxes']
    if args.dsgg_task == 'sgcls':
        losses.append('sub_labels')
    if args.no_update_pair:
        losses = ['relation_labels']
    # if args.dsgg_task == 'predcls':
    #     losses = ['relation_labels']

    criterion = SetCriterionDSGG(num_classes, args.num_queries, matcher=matcher,weight_dict=weight_dict, \
                                 eos_coef=args.eos_coef, losses=losses, args=args)

    criterion.to(device)
    postprocessors = {'dsgg': PostProcessDSGG(args)}

    return model, criterion, postprocessors
