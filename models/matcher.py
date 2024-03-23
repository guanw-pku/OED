import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import pdb

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcherDSGG(nn.Module):
    def __init__(self, cost_obj_class: float = 1, cost_rel_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1, cost_matching: float = 1, use_matching: bool = False):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_rel_class = cost_rel_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_matching = cost_matching
        self.use_matching = use_matching
        assert cost_obj_class != 0 or cost_rel_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_matching != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_obj_logits'].shape[:2]
        out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)
        out_attn_prob = outputs['pred_attn_logits'].flatten(0, 1).softmax(-1)
        out_spatial_prob = outputs['pred_spatial_logits'].flatten(0, 1).sigmoid()
        out_contacting_prob = outputs['pred_contacting_logits'].flatten(0, 1).sigmoid()
        out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
        out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)

        tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])
        tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])
        tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])
        tgt_attn_labels = torch.cat([v['attn_labels'] for v in targets])
        tgt_spatial_labels = torch.cat([v['spatial_labels'] for v in targets])
        tgt_contacting_labels = torch.cat([v['contacting_labels'] for v in targets])
        tgt_attn_labels_permute = tgt_attn_labels.permute(1, 0)
        tgt_spatial_labels_permute = tgt_spatial_labels.permute(1, 0)
        tgt_contacting_labels_permute = tgt_contacting_labels.permute(1, 0)

        cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
        if cost_sub_bbox.shape[1] == 0:
            cost_bbox = cost_sub_bbox
        else:
            cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]
        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                        cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
        if cost_sub_giou.shape[1] == 0:
            cost_giou = cost_sub_giou
        else:
            cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]

        out_attn_prob = out_attn_prob[:, :tgt_attn_labels_permute.shape[0]]
        cost_attn_class = -(out_attn_prob.matmul(tgt_attn_labels_permute) / \
                            (tgt_attn_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_attn_prob).matmul(1 - tgt_attn_labels_permute) / \
                            ((1 - tgt_attn_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
        
        # tgt_attn_labels = torch.where(tgt_attn_labels)[1]
        # cost_attn_class = -out_attn_prob[:, tgt_attn_labels] # TODO: switch to this 
        cost_spatial_class = -(out_spatial_prob.matmul(tgt_spatial_labels_permute) / \
                            (tgt_spatial_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_spatial_prob).matmul(1 - tgt_spatial_labels_permute) / \
                            ((1 - tgt_spatial_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
        cost_contacting_class = -(out_contacting_prob.matmul(tgt_contacting_labels_permute) / \
                            (tgt_contacting_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_contacting_prob).matmul(1 - tgt_contacting_labels_permute) / \
                            ((1 - tgt_contacting_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
        C = self.cost_obj_class * cost_obj_class + self.cost_rel_class * (cost_attn_class + cost_spatial_class + cost_contacting_class) + \
            self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

        if self.use_matching:
            tgt_matching_labels = torch.cat([v['matching_labels'] for v in targets])
            out_matching_prob = outputs['pred_matching_logits'].flatten(0, 1).softmax(-1)
            cost_matching = -out_matching_prob[:, tgt_matching_labels]
            C += self.cost_matching * cost_matching


        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['obj_labels']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return HungarianMatcherDSGG(cost_obj_class=args.set_cost_obj_class, cost_rel_class=args.set_cost_rel_class,
                               cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, 
                               cost_matching=args.set_cost_matching, use_matching=args.use_matching)

