import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from util.misc import intersect_2d, argsort_desc
from util.box_ops import box_iou, box_cxcywh_to_xyxy
import pdb

class BasicSceneGraphEvaluator:
    def __init__(self, mode, iou_threshold=0.5, save_file="tmp", constraint=False, semithreshold=None, error_analysis=False, nms_filter=False):
        self.result_dict = {}
        self.mode = mode
        self.num_rel = 26
        self.num_attn = 3
        self.num_spatial = 6
        self.num_contact = 17
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}  
        self.result_dict[self.mode + '_pair_recall'] = {10: [], 20: [], 50: [], 100: []}  
        self.result_dict[self.mode + '_mean_recall_collect'] = {10: [[] for i in range(self.num_rel)], 20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.constraint = constraint # semi constraint if True
        self.iou_threshold = iou_threshold
        self.semithreshold = semithreshold
        self.tot_all_predicates = 26
        self.save_file = save_file
        with open(self.save_file, "w") as f:
            f.write("Begin training\n")

        self.print_error_case = 20
        self.print_list = []
        self.miss_gt = 0
        self.wrong_rel = 0

        self.nms_filter = nms_filter
        self.print_cls_rec1 = False # other way
        self.print_cls_rec2 = False # tempura way
        self.RELATION_CLASSES = ("looking_at", "not_looking_at", "unsure", "above", "beneath",
                    "in_front_of", "behind", "on_the_side_of", "in", "carrying",
                    "covered_by", "drinking_from", "eating", "have_it_on_the_back",
                    "holding", "leaning_on", "lying_on", "not_contacting",
                    "other_relationship", "sitting_on", "standing_on", "touching",
                    "twisting", "wearing", "wiping", "writing_on")

        self.error_mat = None
        self.error_analysis = error_analysis and self.mode == 'sgdet'
        if self.error_analysis:
            self.error_mat = np.zeros((self.num_rel, 5+1), dtype=int) # 5 error type, 1 for total error number
        
        self.save_cases = 0
        self.easy_bad_case = []
        self.hard_bad_case = []

    def reset_result(self):
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}  
        self.result_dict[self.mode + '_pair_recall'] = {10: [], 20: [], 50: [], 100: []}  
        self.result_dict[self.mode + '_mean_recall_collect'] = {10: [[] for i in range(self.num_rel)], 20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}

    def print_stats(self):
        # print('--------------------------------------')
        # print('miss_gt:', self.miss_gt, '; wrong_rel:', self.wrong_rel)
        # print('--------------------------------------')

        stats_dict = {}
        with open(self.save_file, "a") as f:
            f.write('======================' + self.mode + '============================\n')
            print('======================' + self.mode + '============================')
            for k, v in self.result_dict[self.mode + '_recall'].items():
                print('R@%i: %f' % (k, np.mean(v)))
                f.write('R@%i: %f\n' % (k, np.mean(v)))
                stats_dict['R@%i' % k] = np.mean(v)
                # f.print(self.error_mat)
            # for k, v in self.result_dict[self.mode + '_pair_recall'].items():
            #     print('pair R@%i: %f' % (k, np.mean(v)))
            #     stats_dict['pair R@%i' % k] = np.mean(v)
            
            if self.print_cls_rec1:
                print('             ----------- class recall way 1 ----------')
                for k, v in self.result_dict[self.mode + '_mean_recall_collect'].items():
                    sum_recall = 0 # overall_mean_recall
                    class_recall = []
                    for idx in range(self.num_rel):
                        if len(v[idx]) == 0:
                            tmp_recall = 0.0
                        else:
                            tmp_recall = np.mean(v[idx])
                        class_recall.append(tmp_recall)
                        sum_recall += tmp_recall
                    print('Class Mean R@{}: '.format(k), class_recall)
                    print('Overall Mean R@%i: %f' % (k, sum_recall / float(self.num_rel)))
                    f.write('R@%i: %f\n' % (k, sum_recall / float(self.num_rel)))
            
            if self.print_cls_rec2:
                print('             ----------- class recall way 2 ----------')
                for k, v in self.result_dict[self.mode + '_recall'].items():
                    avg = 0
                    per_class_recall = []
                    for idx in range(self.tot_all_predicates):
                        #print(self.result_dict[self.mode + '_recall_hit'][k][idx+1])
                        tmp_avg = float(self.result_dict[self.mode + '_recall_hit'][k][idx]) / float(self.result_dict[self.mode +'_recall_count'] [k][idx] + 1e-10)

                        avg += tmp_avg
                        per_class_recall.append(tmp_avg)
                    
                    print('Class Mean R@{}: '.format(k), per_class_recall)
                    print('mR@%i: %f'% (k,avg/self.tot_all_predicates))
                    
        if self.error_analysis:
            print('----------- error analysis -----------')
            print(self.error_mat)
            pdb.set_trace()         
            
        return stats_dict

    def evaluate_scene_graph(self, gt, pred):
        '''collect the groundtruth and prediction'''
        for idx, frame_gt in enumerate(gt): # gt[i] indices all annotations for the i-th frame
            # generate the ground truth
            gt_boxes = frame_gt['boxes']
            gt_classes = np.array(frame_gt['labels'])
            gt_relations = []
            human_idx = 0
            for obj_idx in range(len(frame_gt['obj_labels'])):
                gt_relations.append([human_idx, obj_idx+1, int(torch.where(frame_gt['attn_labels'][obj_idx])[0])]) # for attention triplet <human-object-predicate>_
                #spatial and contacting relationship could be multiple
                spatial_idxes = torch.where(frame_gt['spatial_labels'][obj_idx])[0]
                for spatial in spatial_idxes:
                    gt_relations.append([obj_idx+1, human_idx, self.num_attn + spatial]) # for spatial triplet <object-human-predicate>
                contact_idxes = torch.where(frame_gt['contacting_labels'][obj_idx])[0]
                for contact in contact_idxes:
                    gt_relations.append([human_idx, obj_idx+1, self.num_attn + self.num_spatial + contact])  # for contact triplet <human-object-predicate>

            gt_entry = {
                'gt_classes': gt_classes,   
                'gt_relations': np.array(gt_relations),
                'gt_boxes': np.array(box_cxcywh_to_xyxy(gt_boxes)),
            }
            if 'img_path' in frame_gt.keys():
                gt_entry['img_path'] = frame_gt['img_path']


            # first part for attention and contact, second for spatial
            rels_i = np.concatenate((pred[idx]['pair_idx'],             #attention
                                     pred[idx]['pair_idx'][:,::-1],     #spatial
                                     pred[idx]['pair_idx']), axis=0)    #contacting
            pred_scores_1 = np.concatenate((pred[idx]['attention_distribution'],
                                            np.zeros([pred[idx]['pair_idx'].shape[0], pred[idx]['spatial_distribution'].shape[1]]),
                                            np.zeros([pred[idx]['pair_idx'].shape[0], pred[idx]['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_2 = np.concatenate((np.zeros([pred[idx]['pair_idx'].shape[0], pred[idx]['attention_distribution'].shape[1]]),
                                            pred[idx]['spatial_distribution'],
                                            np.zeros([pred[idx]['pair_idx'].shape[0], pred[idx]['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_3 = np.concatenate((np.zeros([pred[idx]['pair_idx'].shape[0], pred[idx]['attention_distribution'].shape[1]]),
                                            np.zeros([pred[idx]['pair_idx'].shape[0], pred[idx]['spatial_distribution'].shape[1]]),
                                            pred[idx]['contacting_distribution']), axis=1)
            
            all_rel_scores = np.concatenate((pred[idx]['attention_distribution'],\
                                              pred[idx]['spatial_distribution'],\
                                              pred[idx]['contacting_distribution']), axis=1)
            if self.mode == 'predcls':
                pred_entry = {
                    'pred_boxes': box_cxcywh_to_xyxy(torch.tensor(pred[idx]['target_boxes'])).numpy(),
                    'pred_classes': pred[idx]['target_labels'],
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred[idx]['target_scores'],
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0),

                    'pred_so_labels': pred[idx]['pair_idx'],
                    'pred_so_attn': pred[idx]['attention_distribution'],
                    'pred_so_spatial': pred[idx]['spatial_distribution'],
                    'pred_so_contacting': pred[idx]['contacting_distribution'],
                }
            elif self.mode == 'sgcls':
                pred_entry = {
                    'pred_boxes': box_cxcywh_to_xyxy(torch.tensor(pred[idx]['target_boxes'])).numpy(), # TODO: test with pred_boxes
                    'pred_classes': pred[idx]['pred_labels'],
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred[idx]['pred_scores'],
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            else:
                pred_entry = {
                    'pred_boxes': box_cxcywh_to_xyxy(torch.tensor(pred[idx]['pred_boxes'])).numpy(),
                    'pred_classes': pred[idx]['pred_labels'],
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred[idx]['pred_scores'],
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            self.evaluate_from_dict(gt_entry, pred_entry, self.mode, iou_thresh=self.iou_threshold, method=self.constraint, 
                                    threshold=self.semithreshold, num_rel= self.num_rel, error_mat=self.error_mat, nms_filter=self.nms_filter)

    def evaluate_from_dict(self, gt_entry, pred_entry, mode, method=None, threshold = 0.9, num_rel=26, **kwargs):
        """
        Shortcut to doing evaluate_recall from dict
        :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
        :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
        :param result_dict:
        :param kwargs:
        :return:
        """
        gt_rels = gt_entry['gt_relations']
        gt_boxes = gt_entry['gt_boxes'].astype(float)
        gt_classes = gt_entry['gt_classes']
        # img_path = gt_entry['img_path']

        pred_rel_inds = pred_entry['pred_rel_inds']
        rel_scores = pred_entry['rel_scores']
        pred_boxes = pred_entry['pred_boxes'].astype(float)
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
        if method == 'no':
            obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
            overall_scores = obj_scores_per_rel[:, None] * (rel_scores) # ** 2
            score_inds = argsort_desc(overall_scores) # top100
            if kwargs['nms_filter']:
                score_inds = score_inds[:500]
            else:
                score_inds = score_inds[:100]
            pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
            predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]
        else:
            pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1))) #1+  dont add 1 because no dummy 'no relations'
            predicate_scores = rel_scores.max(1)

        # print(gt_entry['img_path'])
        # pred_to_gt, pred_to_gt_pair, pred_5ples, rel_scores = self.evaluate_recall(
        pred_to_gt, pred_5ples, rel_scores = self.evaluate_recall(
                    gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes,
                    predicate_scores, obj_scores, phrdet= mode=='phrdet',
                    **kwargs)
        for k in self.result_dict[mode + '_recall']:
            match = reduce(np.union1d, pred_to_gt[:k])
            # match_pair = reduce(np.union1d, pred_to_gt_pair[:k])

            recall_hit = [0] * num_rel
            recall_count = [0] * num_rel

            for idx in range(len(match)): 
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += 1

                if (mode + '_recall_hit') not in self.result_dict:
                    self.result_dict[mode + '_recall_hit'] = {}
                if k not in self.result_dict[mode + '_recall_hit']:
                    self.result_dict[mode + '_recall_hit'][k] = [0] * (self.tot_all_predicates)
                self.result_dict[mode + '_recall_hit'][k][int(local_label)] += 1
                #result_dict[mode + '_recall_hit'][k][0] += 1
            
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                if (mode + '_recall_count') not in self.result_dict:
                    self.result_dict[mode + '_recall_count'] = {}
                if k not in self.result_dict[mode + '_recall_count']:
                    self.result_dict[mode + '_recall_count'][k] = [0] * (self.tot_all_predicates)
                self.result_dict[mode + '_recall_count'][k][int(local_label)] += 1


            for n in range(num_rel):
                if recall_count[n] > 0:
                    # if k == 50 and method == 'no' and recall_count[n] > 0 and recall_hit[n] < recall_count[n]:
                    #     print(self.RELATION_CLASSES[n], img_path)
                    #     pdb.set_trace()
                    #     self.print_list.append((self.RELATION_CLASSES[n], img_path))
                    #     self.print_error_case -= 1
                    #     if self.print_error_case == 0:
                    #         for print_item in self.print_list:
                    #             print(print_item)
                    #         pdb.set_trace()
                    self.result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
                


                        
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            # rec_i_pair = float(len(match_pair)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall'][k].append(rec_i)
            # result_dict[mode + '_pair_recall'][k].append(rec_i_pair)

            # if self.constraint == 'no' and k == 10:
            #     if rec_i < 0.83:
            #         pred_so_labels = pred_entry['pred_classes'][pred_entry['pred_so_labels']]
            #         pred_so_attn = pred_entry['pred_so_attn']
            #         pred_so_spatial = pred_entry['pred_so_spatial']
            #         pred_so_contacting = pred_entry['pred_so_contacting']
                    
            #         gt_classes = gt_entry['gt_classes']
            #         gt_relations = gt_entry['gt_relations']
            #         gt_triplet_labels = np.stack([gt_classes[gt_relations[:, 0]], gt_classes[gt_relations[:, 1]], gt_relations[:, -1]], axis=1)

            #         bad_case = {'pred_so_labels': pred_so_labels, 'pred_so_attn': pred_so_attn, 'pred_so_spatial': pred_so_spatial, 
            #                     'pred_so_contacting': pred_so_contacting, 'gt_triplet_labels': gt_triplet_labels, 'rec': rec_i}

            #         if rec_i > 0.5:
            #             self.easy_bad_case.append(bad_case)
            #         else:
            #             self.hard_bad_case.append(bad_case)
                    
            #         self.save_cases += 1
            #         if self.save_cases == 100:
            #             pdb.set_trace()

        return pred_to_gt, pred_5ples, rel_scores

###########################
    def evaluate_recall(self, gt_rels, gt_boxes, gt_classes,
                        pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                        iou_thresh=0.5, phrdet=False, error_mat=None, nms_filter=False):
        """
        Evaluates the recall
        :param gt_rels: [#gt_rel, 3] array of GT relations
        :param gt_boxes: [#gt_box, 4] array of GT boxes
        :param gt_classes: [#gt_box] array of GT classes
        :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                        and refer to IDs in pred classes / pred boxes
                        (id0, id1, rel)
        :param pred_boxes:  [#pred_box, 4] array of pred boxes
        :param pred_classes: [#pred_box] array of predicted classes for these boxes
        :return: pred_to_gt: Matching from predicate to GT
                pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
                rel_scores: [cls_0score, cls1_score, relscore]
                    """
        if pred_rels.size == 0:
            return [[]], np.zeros((0,5)), np.zeros(0)

        num_gt_boxes = gt_boxes.shape[0]
        num_gt_relations = gt_rels.shape[0]
        assert num_gt_relations != 0

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                    gt_rels[:, :2],
                                                    gt_classes,
                                                    gt_boxes)
        num_boxes = pred_boxes.shape[0]
        if pred_rels[:,:2].max() >= pred_classes.shape[0]:
            pdb.set_trace()

        # Exclude self rels
        # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])
        #assert np.all(pred_rels[:,2] > 0)
        
        pred_triplets, pred_triplet_boxes, relation_scores = \
            _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                    rel_scores, cls_scores)

        sorted_scores = relation_scores.prod(1)
        pred_triplets = pred_triplets[sorted_scores.argsort()[::-1],:]
        pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1],:]
        relation_scores = relation_scores[sorted_scores.argsort()[::-1],:]
        scores_overall = relation_scores.prod(1)

        if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
            print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
            # raise ValueError("Somehow the relations werent sorted properly")

        # NMS postprocess
        if nms_filter:
            pred_triplets, pred_triplet_boxes, filterd_idx = triplet_nms_filter(pred_triplets, pred_triplet_boxes, scores_overall)
            pred_rels = pred_rels[filterd_idx]

        # Compute recall. It's most efficient to match once and then do recall after
        # match prediction to ground truth, many to many.
        pred_to_gt = self._compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thresh,
            phrdet=phrdet
        )

        # # ----------- pair detection ------------
        # pred_to_gt_pair = self._compute_pred_matches(gt_triplets[:, 0::2], pred_triplets[:, 0::2], gt_triplet_boxes, pred_triplet_boxes,
        #     iou_thresh, match_pair=True
        # )
        # # ---------------------------------------

        # match10 = reduce(np.union1d, pred_to_gt[:10])
        # match20 = reduce(np.union1d, pred_to_gt[:20])
        # match50 = reduce(np.union1d, pred_to_gt[:50])
        # matchall = reduce(np.union1d, pred_to_gt)
        # print('match_10:', len(match10), '; match_20:', len(match20), 'match_50:', len(match50), '; match_all:', len(matchall), '; num_gt:', num_gt_relations)

        # if len(match50) < len(gt_triplets):
        #     print('match_50: ', len(match50), '; match_all: ', len(matchall), '; num_gt: ', num_gt_relations)
        #     pdb.set_trace()
        #     _ = self._compute_pred_matches(
        #         gt_triplets,
        #         pred_triplets,
        #         gt_triplet_boxes,
        #         pred_triplet_boxes,
        #         iou_thresh,
        #         phrdet=phrdet,
        #         is_pdb=False,
        #         relation_scores=relation_scores
        #     )
        # Error analysis
        if error_mat is not None:
            error_analysis(pred_to_gt, gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes, error_mat)

        # Contains some extra stuff for visualization. Not needed.
        pred_5ples = np.column_stack((
            pred_rels[:,:2],
            pred_triplets[:, [0, 2, 1]],
        ))

        return pred_to_gt, pred_5ples, relation_scores
        # return pred_to_gt, pred_to_gt_pair, pred_5ples, relation_scores
    

    def _compute_pred_matches(self, gt_triplets, pred_triplets,
                    gt_boxes, pred_boxes, iou_thresh, phrdet=False, is_pdb=False, match_pair=False, relation_scores=None):
        """
        Given a set of predicted triplets, return the list of matching GT's for each of the
        given predictions
        :param gt_triplets:
        :param pred_triplets:
        :param gt_boxes:
        :param pred_boxes:
        :param iou_thresh:
        :return:
        """
        # This performs a matrix multiplication-esque thing between the two arrays
        # Instead of summing, we want the equality, so we reduce in that way
        # The rows correspond to GT triplets, columns to pred triplets
        if is_pdb:
            pdb.set_trace()

        keeps = intersect_2d(gt_triplets, pred_triplets)
        gt_has_match = keeps.any(1)
        # pdb.set_trace()
        pred_to_gt = [[] for x in range(pred_boxes.shape[0])]

        for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                            gt_boxes[gt_has_match],
                                            keeps[gt_has_match]):
            boxes = pred_boxes[keep_inds]
            if phrdet:
                # Evaluate where the union box > 0.5
                gt_box_union = gt_box.reshape((2, 4))
                gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

                box_union = boxes.reshape((-1, 2, 4))
                box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

                inds = box_iou(gt_box_union[None], box_union)[0] >= iou_thresh

            else: # pass for sgcls and predcls
                sub_iou = box_iou(torch.tensor(gt_box[None,:4]), torch.tensor(boxes[:, :4]))[0]
                obj_iou = box_iou(torch.tensor(gt_box[None,4:]), torch.tensor(boxes[:, 4:]))[0]

                inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)
                # if not match_pair:
                #     inds_2 = (torch.max(sub_iou, obj_iou) >= 0.5) & (torch.min(sub_iou, obj_iou) >= 0.3)
                #     if inds.sum() == 0:
                #         obj_label_keep_inds = intersect_2d(gt_triplets[gt_ind][None, :], pred_triplets, obj_label=True)[0]
                #         boxes = pred_boxes[obj_label_keep_inds]
                #         sub_iou = box_iou(torch.tensor(gt_box[None,:4]), torch.tensor(boxes[:, :4]))[0]
                #         obj_iou = box_iou(torch.tensor(gt_box[None,4:]), torch.tensor(boxes[:, 4:]))[0]
                #         inds_3 = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)
                #         if inds_3.sum() == 0:
                #             inds = inds_2
                #         else:
                #             self.wrong_rel = self.wrong_rel + 1

            inds = np.array(inds)[0]
            for i in np.where(keep_inds)[0][inds]:
                pred_to_gt[i].append(int(gt_ind))

        # hit_gt_idx_list = reduce(np.union1d, pred_to_gt)
        # if not match_pair:
        #     self.miss_gt += len(gt_triplets) - len(hit_gt_idx_list)

        return pred_to_gt


def triplet_nms_filter(pred_triplets, pred_triplet_boxes, scores_overall):
    all_triplets = {}
    for pred_idx in range(len(pred_triplets)):
        triplet = str(pred_triplets[pred_idx])
        if triplet not in all_triplets:
            all_triplets[triplet] = {'triplet_boxes': [], 'scores': [], 'indexes': []}
        all_triplets[triplet]['triplet_boxes'].append(pred_triplet_boxes[pred_idx])
        all_triplets[triplet]['scores'].append(scores_overall[pred_idx])
        all_triplets[triplet]['indexes'].append(pred_idx)

    all_keep_inds = []
    for triplet, values in all_triplets.items():
        triplet_boxes, scores = values['triplet_boxes'], values['scores']
        keep_inds = pairwise_nms(np.array(triplet_boxes), np.array(scores))
        keep_inds = list(np.array(values['indexes'])[keep_inds])
        all_keep_inds.extend(keep_inds)
    all_keep_inds.sort()
    
    return pred_triplets[all_keep_inds], pred_triplet_boxes[all_keep_inds], all_keep_inds


def pairwise_nms(triplet_boxes, scores):
    nms_alpha, nms_beta, thres_nms = 1.0, 1.0, 0.9
    # print(nms_alpha, nms_beta, thres_nms)
    
    subs = triplet_boxes[:, :4]
    objs = triplet_boxes[:, 4:]

    sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
    ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

    sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
    obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

    order = scores.argsort()[::-1]

    keep_inds = []
    while order.size > 0:
        i = order[0]
        keep_inds.append(i)

        sxx1 = np.maximum(sx1[i], sx1[order[1:]])
        syy1 = np.maximum(sy1[i], sy1[order[1:]])
        sxx2 = np.minimum(sx2[i], sx2[order[1:]])
        syy2 = np.minimum(sy2[i], sy2[order[1:]])

        sw = np.maximum(0.0, sxx2 - sxx1 + 1)
        sh = np.maximum(0.0, syy2 - syy1 + 1)
        sub_inter = sw * sh
        sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

        oxx1 = np.maximum(ox1[i], ox1[order[1:]])
        oyy1 = np.maximum(oy1[i], oy1[order[1:]])
        oxx2 = np.minimum(ox2[i], ox2[order[1:]])
        oyy2 = np.minimum(oy2[i], oy2[order[1:]])

        ow = np.maximum(0.0, oxx2 - oxx1 + 1)
        oh = np.maximum(0.0, oyy2 - oyy1 + 1)
        obj_inter = ow * oh
        obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

        ovr = np.power(sub_inter/sub_union, nms_alpha) * np.power(obj_inter / obj_union, nms_beta)
        inds = np.where(ovr <= thres_nms)[0]

        order = order[inds + 1]
    return keep_inds


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])
    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def error_analysis(pred_to_gt, gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes, error_mat):
    num_gt = len(gt_triplets)
    num_pred = len(pred_triplets)
    hit_gt_idx_list = reduce(np.union1d, pred_to_gt)
    for i in range(num_gt):
        if i in hit_gt_idx_list:
            continue
        gt_sub_cat = gt_triplets[i][0]
        gt_rel_cat = gt_triplets[i][1]
        gt_obj_cat = gt_triplets[i][2]
        gt_sub_box = gt_triplet_boxes[i][None, :4]
        gt_obj_box = gt_triplet_boxes[i][None, 4:]
        if gt_sub_cat > 1:
            gt_sub_cat, gt_obj_cat = gt_obj_cat, gt_sub_cat
            gt_sub_box, gt_obj_box = gt_obj_box, gt_sub_box
            assert gt_sub_cat == 1
        error_mat[int(gt_rel_cat)][-1] += 1
        
        cur_status = 0
        for j in range(num_pred):
            pred_sub_cat = pred_triplets[j][0]
            pred_rel_cat = pred_triplets[j][1]
            pred_obj_cat = pred_triplets[j][2]
            pred_sub_box = pred_triplet_boxes[j][None, :4]
            pred_obj_box = pred_triplet_boxes[j][None, 4:]
            if pred_sub_cat > 1:
                pred_sub_cat, pred_obj_cat = pred_obj_cat, pred_sub_cat
                pred_sub_box, pred_obj_box = pred_obj_box, pred_sub_box
                assert pred_sub_cat == 1
            sub_iou = bbox_overlaps(gt_sub_box, pred_sub_box)[0]
            obj_iou = bbox_overlaps(gt_obj_box, pred_obj_box)[0]

            if sub_iou < 0.1:
                cur_status = max(cur_status, 0)
            elif sub_iou < 0.5:
                cur_status = max(cur_status, 1)
            elif obj_iou < 0.5:
                cur_status = max(cur_status, 2)
            elif pred_obj_cat != gt_obj_cat:
                cur_status = max(cur_status, 3)
            else:
                cur_status = max(cur_status, 4)

        error_mat[int(gt_rel_cat)][cur_status] += 1
    
    return error_mat

def print_case(case_dict):
    print('recall: ', case_dict['rec'])
    print('-----------------')
    print('gt_triplet:', case_dict['gt_triplet_labels'])
    print('-----------------')
    print('pred_so_lables: ', case_dict['pred_so_labels'])
    print('pred_so_attn: ', case_dict['pred_so_attn'])
    print('pred_so_spatial: ', case_dict['pred_so_spatial'])
    print('pred_so_contacting: ', case_dict['pred_so_contacting'])
    print('-----------------')
