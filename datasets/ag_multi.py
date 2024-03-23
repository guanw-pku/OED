# Modified by Lu He
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from .coco_video_parser import CocoVID
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_multi as T
from torch.utils.data.dataset import ConcatDataset
import random
import pdb

class DSGGDataset(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, interval1, interval2, num_ref_frames=3, \
                 is_train=True, filter_key_img=True, cache_mode=False, local_rank=0, local_size=1, 
                 seq_sort=False, return_ref_targets=False):
        super(DSGGDataset, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(transforms)
        self.ann_file = ann_file
        self.num_ref_frames = num_ref_frames
        self.cocovid = CocoVID(self.ann_file)
        self.is_train = is_train
        self.filter_key_img = filter_key_img
        self.interval1 = interval1
        self.interval2 = interval2
        self.seq_sort = seq_sort
        self.load_all_target = True
        self.return_ref_targets = return_ref_targets

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # img_id == idx + 1
        imgs = []
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        video_id = img_info['video_id']
        img = self.get_image(path)

        target = {'image_id': img_id, 'annotations': target}

        img, target = self.prepare(img, target)
        if not self.is_train:
            target['img_path'] = path
        imgs.append(img)

        if self.return_ref_targets:
            targets = []
            targets.append(target)
        
        img_ids = self.cocovid.get_img_ids_from_vid(video_id)
        ref_img_ids = []
        if self.is_train:
            interval = self.interval1
            left = max(img_ids[0], img_id - interval)
            right = min(img_ids[-1], img_id + interval)
            sample_range = list(range(left, right + 1))
            if self.filter_key_img and img_id in sample_range:
                sample_range.remove(img_id)
            while len(sample_range) < self.num_ref_frames:
                print('sample range:', sample_range)
                sample_range.extend(sample_range)
            ref_img_ids = random.sample(sample_range, self.num_ref_frames)

        else:
            interval = self.interval2
            left = max(img_ids[0], img_id - interval)
            right = min(img_ids[-1], img_id + interval)
            ref_range = list(range(left, right + 1))
            if self.filter_key_img and img_id in ref_range:
                ref_range.remove(img_id)
            step = 1
            if len(ref_range) > self.num_ref_frames:
                step = (len(ref_range) - 1) // (self.num_ref_frames - 1)
            for i in ref_range[::step]:
                ref_img_ids.append(i)
            while len(ref_img_ids) < self.num_ref_frames:
                print('reference frames: ', ref_img_ids)
                ref_img_ids.extend(ref_img_ids) 
            ref_img_ids = ref_img_ids[:self.num_ref_frames]

            # interval = self.interval2
            # # left = max(img_ids[0], img_id - interval)
            # left = img_id - interval
            # # right = min(img_ids[-1], img_id + interval)
            # right = img_id + interval
            # step = (right - left) // (self.num_ref_frames - 1)

            # for i in range(left, right + 1, step):
            #     ref_img_ids.append(min(max(img_ids[0], i), img_ids[-1]))
            # if self.filter_key_img and img_id in ref_img_ids:
            #     ref_img_ids.remove(img_id)
            # while len(ref_img_ids) < self.num_ref_frames:
            #     print('reference frames: ', ref_img_ids)
            #     ref_img_ids.extend(ref_img_ids) 
            # ref_img_ids = ref_img_ids[:self.num_ref_frames]    

        for ref_img_id in ref_img_ids:
            ref_ann_ids = coco.getAnnIds(imgIds=ref_img_id)
            ref_target = coco.loadAnns(ref_ann_ids)
            ref_img_info = coco.loadImgs(ref_img_id)[0] 
            ref_img_path = ref_img_info['file_name']
            ref_img = self.get_image(ref_img_path)
            imgs.append(ref_img)
            
            if self.return_ref_targets:
                ref_target = {'image_id': ref_img_id, 'annotations': ref_target}
                _, ref_target = self.prepare(ref_img, ref_target)
                targets.append(ref_target)

        if self.seq_sort:
            ref_img_ids = [img_id] + ref_img_ids
            ids_idx_pair = sorted(enumerate(ref_img_ids), key=lambda x: x[1])
            sorted_idx = [i for i, _ in ids_idx_pair]
            imgs = [imgs[i] for i in sorted_idx]
            if self.return_ref_targets:
                targets = [targets[i] for i in sorted_idx]
            targets[0]['cur_idx'] = torch.as_tensor([sorted_idx.index(0)])
        else:
            if self.return_ref_targets:
                targets[0]['cur_idx'] = torch.as_tensor([0])
            else:
                target['cur_idx'] = torch.as_tensor([0])

        if self.return_ref_targets:
            if self._transforms is not None:
                imgs, targets = self._transforms(imgs, targets)
            targets = self.post_process_target(targets)
            return torch.cat(imgs, dim=0), targets
        else:
            if self._transforms is not None:
                imgs, target = self._transforms(imgs, target)
            target = self.post_process_target(target)
            return torch.cat(imgs, dim=0), target
    
    def post_process_target(self, target):

        if type(target) == list:
            targets = []
            for tgt in target:
                tgt = self.post_process_target(tgt)
                targets.append(tgt)
            return targets

        classes = target['labels']
        num_objs = len(classes) - 1

        if num_objs == 0 or (classes == 1).sum() == 0:
            target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
            target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        else:
            target['obj_labels'] = target['labels'][1:]
            target['sub_boxes'] = target['boxes'][0].repeat((num_objs, 1))
            target['obj_boxes'] = target['boxes'][1:]
        return target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        # pdb.set_trace()
        # img_path = target['img_path']

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0 or obj['iscrowd'] == -1]
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        attn_labels, spatial_labels, contacting_labels = [], [], []
        for oid in range(1, len(anno)):
            obj = anno[oid]
            attn_obj_label = torch.zeros(3)
            spatial_obj_label = torch.zeros(6)
            contacting_obj_label = torch.zeros(17)
            attn_obj_label[obj['attention_rel']] = 1
            spatial_obj_label[torch.tensor(obj['spatial_rel']) - 3] = 1
            contacting_obj_label[torch.tensor(obj['contact_rel']) - 9] = 1
            attn_labels.append(attn_obj_label)
            spatial_labels.append(spatial_obj_label)
            contacting_labels.append(contacting_obj_label)
        attn_labels = torch.stack(attn_labels, dim=0)
        spatial_labels = torch.stack(spatial_labels, dim=0)
        contacting_labels = torch.stack(contacting_labels, dim=0)
        
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])[keep]
        area = torch.tensor([obj["area"] for obj in anno])[keep]
        attn_labels = attn_labels[keep[1:]]
        spatial_labels = spatial_labels[keep[1:]]
        contacting_labels = contacting_labels[keep[1:]]

        target = {}
        target["orig_size"] = torch.as_tensor([int(h), int(w)]) # h, w!
        target["size"] = torch.as_tensor([int(h), int(w)])
        target['boxes'] = boxes
        target['labels'] = classes
        target["iscrowd"] = iscrowd
        target["area"] = area
        
        num_objs = len(classes) - 1
        if num_objs == 0 or (classes == 1).sum() == 0:
            target['attn_labels'] = torch.zeros((0, 3), dtype=torch.float32)
            target['spatial_labels'] = torch.zeros((0, 6), dtype=torch.float32)
            target['contacting_labels'] = torch.zeros((0, 17), dtype=torch.float32)
        else:
            target['attn_labels'] = attn_labels
            target['spatial_labels'] = spatial_labels
            target['contacting_labels'] = contacting_labels

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # T.Normalize([0.4196, 0.3736, 0.3451], [0.2859, 0.2810, 0.2784])
    ])

    if image_set == 'train':
        return T.Compose([
            # T.RandomHorizontalFlip(),
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.ag_path)
    assert root.exists(), f'provided Action Genome path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root, root / "annotations" / 'ag_train_coco_style.json'),
        "val": (root, root / "annotations" / 'ag_test_coco_style.json'),
    }
    img_folder, anno_file = PATHS[image_set]
    dataset = DSGGDataset(img_folder, anno_file, transforms=make_coco_transforms(image_set), is_train=(not args.eval), 
                          interval1=args.interval1, interval2=args.interval2, num_ref_frames=args.num_ref_frames, 
                          cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), 
                          seq_sort=args.seq_sort, return_ref_targets=args.dsgg_task in ['sgcls', 'predcls'] or args.use_matched_query)

    return dataset
