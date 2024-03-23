from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import numpy as np
from queue import Queue
import math
import pdb

from .backbone import build_backbone
from .matcher import build_matcher
from .cdn_single_sgdet import build_cdn


class CDNDSGG(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_queries, aux_loss=False, args=None):
        super().__init__()
        # task determines whether object pair detection results are given
        self.dsgg_task = args.dsgg_task

        # model parameters
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.attn_class_embed = nn.Linear(hidden_dim, args.num_attn_classes + 1) # add 1 for background
        self.spatial_class_embed = nn.Linear(hidden_dim, args.num_spatial_classes)
        self.contacting_class_embed = nn.Linear(hidden_dim, args.num_contacting_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

        # hyperparameters
        self.aux_loss = aux_loss
        self.use_matching = args.use_matching
        self.dec_layers_hopd = args.dec_layers_hopd
        self.dec_layers_interaction = args.dec_layers_interaction

    def forward(self, samples: NestedTensor, targets=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        # hopd_out, interaction_decoder_out = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[:2]
        hopd_out, interaction_decoder_out, ins_attn_weight, rel_attn_weight = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[:4]

        outputs_sub_coord = self.sub_bbox_embed(hopd_out).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hopd_out).sigmoid()
        outputs_obj_class = self.obj_class_embed(hopd_out)

        outputs_attn_class = self.attn_class_embed(interaction_decoder_out)
        outputs_spatial_class = self.spatial_class_embed(interaction_decoder_out)
        outputs_contacting_class = self.contacting_class_embed(interaction_decoder_out)
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
               'pred_attn_logits': outputs_attn_class[-1], 'pred_spatial_logits': outputs_spatial_class[-1], 'pred_contacting_logits': outputs_contacting_class[-1],
               'ins_attn_weight': ins_attn_weight, 'rel_attn_weight': rel_attn_weight}

        if self.aux_loss:                      
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_sub_coord, outputs_obj_coord,
                                                    outputs_attn_class, outputs_spatial_class,
                                                    outputs_contacting_class)

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

        self.alpha = args.alpha


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

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        obj_weights = self.empty_weight
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, obj_weights)
        # loss_obj_ce = F.cross_entropy(src_logits[idx], target_classes_o, obj_weights)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
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
        # target_attn_classes_o = torch.cat([target_attn_classes_o, torch.zeros_like(target_attn_classes_o[:, 0:1])], dim=-1)
        target_spatial_classes_o = torch.cat([t['spatial_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_contacting_classes_o = torch.cat([t['contacting_labels'][J] for t, (_, J) in zip(targets, indices)])
        # target_attn_classes = torch.zeros_like(outputs['pred_attn_logits'])
        # target_spatial_classes = torch.zeros_like(outputs['pred_spatial_logits'])
        # target_contacting_classes = torch.zeros_like(outputs['pred_contacting_logits'])
        # target_attn_classes[idx] = target_attn_classes_o
        # target_spatial_classes[idx] = target_spatial_classes_o
        # target_contacting_classes[idx] = target_contacting_classes_o

        # target_attn_classes = target_attn_classes.reshape(-1, num_attn_rel + 1)
        # target_spatial_classes = target_spatial_classes.reshape(-1, num_spatial_rel)
        # target_contacting_classes = target_contacting_classes.reshape(-1, num_contacting_rel)

        # # focal loss to balance positive/negative
        # pos_attn_inds = target_attn_classes.eq(1).float()
        # unmatched_flag = pos_attn_inds.sum(-1) == 0
        
        # target_attn_classes[unmatched_flag, -1] = 1

        ## only select matched query to calculate loss
        sel_idx = idx[0] * outputs['pred_attn_logits'].shape[1] + idx[1]
        attn_logits = attn_logits[sel_idx]
        spatial_probs = spatial_probs[sel_idx]
        contacting_probs = contacting_probs[sel_idx]
        target_attn_classes = target_attn_classes_o
        target_spatial_classes = target_spatial_classes_o
        target_contacting_classes = target_contacting_classes_o
        ## -----------------------------------

        target_attn_labels = torch.where(target_attn_classes)[1]
        loss_attn_ce = F.cross_entropy(attn_logits, target_attn_labels)
        loss_spatial_ce = self._neg_loss(spatial_probs, target_spatial_classes, alpha=self.alpha)
        loss_contacting_ce = self._neg_loss(contacting_probs, target_contacting_classes, alpha=self.alpha)
        
        losses = {'loss_attn_ce': loss_attn_ce, 'loss_spatial_ce': loss_spatial_ce, 'loss_contacting_ce': loss_contacting_ce}

        return losses


    # def loss_relation_labels(self, outputs, targets, indices, num_interactions):
    #     num_attn_rel, num_spatial_rel, num_contacting_rel = 3, 6, 17
    #     attn_logits = outputs['pred_attn_logits'].reshape(-1, num_attn_rel + 1)
    #     spatial_logits = outputs['pred_spatial_logits'].reshape(-1, num_spatial_rel)
    #     contacting_logits = outputs['pred_contacting_logits'].reshape(-1, num_contacting_rel)
    #     attn_probs = attn_logits.softmax(dim=-1)
    #     spatial_probs = spatial_logits.sigmoid()
    #     contacting_probs = contacting_logits.sigmoid()

    #     idx = self._get_src_permutation_idx(indices)
    #     target_attn_classes_o = torch.cat([t['attn_labels'][J] for t, (_, J) in zip(targets, indices)])
    #     target_attn_classes_o = torch.cat([target_attn_classes_o, torch.zeros_like(target_attn_classes_o[:, 0:1])], dim=-1)
    #     target_spatial_classes_o = torch.cat([t['spatial_labels'][J] for t, (_, J) in zip(targets, indices)])
    #     target_contacting_classes_o = torch.cat([t['contacting_labels'][J] for t, (_, J) in zip(targets, indices)])
    #     target_attn_classes = torch.zeros_like(outputs['pred_attn_logits'])
    #     target_spatial_classes = torch.zeros_like(outputs['pred_spatial_logits'])
    #     target_contacting_classes = torch.zeros_like(outputs['pred_contacting_logits'])
    #     target_attn_classes[idx] = target_attn_classes_o
    #     target_spatial_classes[idx] = target_spatial_classes_o
    #     target_contacting_classes[idx] = target_contacting_classes_o

    #     target_attn_classes = target_attn_classes.reshape(-1, num_attn_rel + 1)
    #     target_spatial_classes = target_spatial_classes.reshape(-1, num_spatial_rel)
    #     target_contacting_classes = target_contacting_classes.reshape(-1, num_contacting_rel)

    #     pos_attn_inds = target_attn_classes.eq(1).float()
    #     unmatched_flag = pos_attn_inds.sum(-1) == 0
        
    #     # Assign the last class to the ground truth of nagetive samples for attention relation classification
    #     target_attn_classes[unmatched_flag, -1] = 1

    #     # Cross entropy loss for attention relation
    #     target_attn_labels = torch.where(target_attn_classes)[1]
    #     loss_attn_ce = F.cross_entropy(attn_logits, target_attn_labels)

    #     # Focal loss to balance positive/negative
    #     loss_spatial_ce = self._neg_loss(spatial_probs, target_spatial_classes, alpha=self.alpha)
    #     loss_contacting_ce = self._neg_loss(contacting_probs, target_contacting_classes, alpha=self.alpha)
        
    #     losses = {'loss_attn_ce': loss_attn_ce, 'loss_spatial_ce': loss_spatial_ce, 'loss_contacting_ce': loss_contacting_ce}

    #     return losses


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

    def loss_matching_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_matching_logits' in outputs
        src_logits = outputs['pred_matching_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['matching_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_matching = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_matching': loss_matching}

        if log:
            losses['matching_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
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
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'relation_labels': self.loss_relation_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'matching_labels': self.loss_matching_labels
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)

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
                indices = self.matcher(aux_outputs, targets)
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

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']
        out_attn_logits = outputs['pred_attn_logits']
        out_spatial_logits = outputs['pred_spatial_logits']
        out_contacting_logits = outputs['pred_contacting_logits']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        # attn_probs = out_attn_logits.softmax(-1)[..., :-1]
        attn_probs = out_attn_logits[..., :-1].softmax(-1)
        spatial_probs = out_spatial_logits.sigmoid()
        contacting_probs = out_contacting_logits.sigmoid()
        out_boxes = torch.cat([out_sub_boxes, out_obj_boxes], dim=1)

        results = []
        for index in range(len(target_sizes)):
            frame_pred = {}
            frame_pred['pred_scores'] = torch.cat([torch.ones(out_sub_boxes.shape[1]), obj_scores[index].cpu()]).numpy()
            frame_pred['pred_labels'] = torch.cat([torch.ones(out_sub_boxes.shape[1]), obj_labels[index].cpu()]).numpy()
            frame_pred['pred_boxes'] = out_boxes[index].cpu().numpy()
            frame_pred['pair_idx'] = torch.cat([torch.arange(out_sub_boxes.shape[1])[:, None], \
                                                torch.arange(out_sub_boxes.shape[1], 2 * out_sub_boxes.shape[1])[:, None]], dim=1).cpu().numpy()
            frame_pred['attention_distribution'] = attn_probs[index].cpu().numpy()
            frame_pred['spatial_distribution'] = spatial_probs[index].cpu().numpy()
            frame_pred['contacting_distribution'] = contacting_probs[index].cpu().numpy()

            results.append(frame_pred)

        return results


def build(args):

    if 'ag' in args.dataset_file:
        num_classes = 36 + 1

    device = torch.device(args.device)

    backbone = build_backbone(args)

    cdn = build_cdn(args)

    model = CDNDSGG(
        backbone,
        cdn,
        num_obj_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_attn_ce'] = args.rel_loss_coef
    weight_dict['loss_spatial_ce'] = args.rel_loss_coef
    weight_dict['loss_contacting_ce'] = args.rel_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.use_matching:
        weight_dict['loss_matching'] = args.matching_loss_coef

    if args.aux_loss:
        min_dec_layers_num = min(args.dec_layers_hopd, args.dec_layers_interaction)
        aux_weight_dict = {}
        for i in range(min_dec_layers_num - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['obj_labels', 'relation_labels', 'sub_obj_boxes', 'obj_cardinality']
    if args.use_matching:
        losses.append('matching_labels')

    criterion = SetCriterionDSGG(num_classes, args.num_queries, matcher=matcher,weight_dict=weight_dict, \
                                 eos_coef=args.eos_coef, losses=losses, args=args)

    criterion.to(device)
    postprocessors = {'dsgg': PostProcessDSGG(args)}

    return model, criterion, postprocessors
