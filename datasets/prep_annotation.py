import os
import json
import pdb
import pickle
from tqdm import tqdm

CLASSES = ("person", "bag", "bed", "blanket", "book", "box", "broom", "chair", 
           "closet/cabinet", "clothes", "cup/glass/bottle", "dish", "door", 
           "doorknob", "doorway", "floor", "food", "groceries", "laptop", 
           "light", "medicine", "mirror", "paper/notebook", "phone/camera", 
           "picture", "pillow", "refrigerator", "sandwich", "shelf", "shoe", "sofa/couch",
           "table", "television", "towel", "vacuum", "window")

RELATION_CLASSES = ("looking_at", "not_looking_at", "unsure", "above", "beneath",
                    "in_front_of", "behind", "on_the_side_of", "in", "carrying",
                    "covered_by", "drinking_from", "eating", "have_it_on_the_back",
                    "holding", "leaning_on", "lying_on", "not_contacting",
                    "other_relationship", "sitting_on", "standing_on", "touching",
                    "twisting", "wearing", "wiping", "writing_on")


cat_id_maps = {}
for k, v in enumerate(CLASSES, 1): # index from 1
    cat_id_maps[v] = k

rel_cat_id_maps = {}
for k, v in enumerate(RELATION_CLASSES, 0): # index from 0
    rel_cat_id_maps[v] = k

def trans_annotations(root_path, mode):
    with open(os.path.join(root_path, 'annotations/person_bbox.pkl'), 'rb') as f:
        person_bbox = pickle.load(f)
    f.close()
    with open(os.path.join(root_path, 'annotations/object_bbox_and_relationship.pkl'), 'rb') as f:
        object_bbox = pickle.load(f)
    f.close()

    # collect valid frames
    video_dict = {}
    for i in person_bbox.keys():
        if object_bbox[i][0]['metadata']['set'] == mode: # train or test
            frame_valid = False
            for j in object_bbox[i]: # the frame is valid if there is visible bbox
                if j['visible']:
                    frame_valid = True
                    break
            if frame_valid:
                video_name, frame_num = i.split('/')
                if video_name in video_dict.keys():
                    video_dict[video_name].append(i)
                else:
                    video_dict[video_name] = [i]

    video_list = []
    video_size = [] # (w,h)
    gt_annotations = []
    non_gt_human_nums = 0
    non_person_video = 0
    one_frame_video = 0
    valid_nums = 0

    for i in video_dict.keys():
        video = []
        gt_annotation_video = []
        for j in video_dict[i]:
            if person_bbox[j]['bbox'].shape[0] == 0:
                non_gt_human_nums += 1
                continue
            else:
                video.append(j)
                valid_nums += 1
            
            p_box = person_bbox[j]['bbox'][0].tolist()
            gt_annotation_frame = [{'person_bbox': [p_box[0], p_box[1], p_box[2] - p_box[0], p_box[3] - p_box[1]], "frame": j}] # xyxy2uxuywh
            # each frames's objects and human
            for k in object_bbox[j]:
                if k['visible']:
                    assert k['bbox'] != None, 'warning! The object is visible without bbox'
                    k['class'] = CLASSES.index(k['class']) + 1 # object index from 1
                    k['bbox'] = [k['bbox'][0], k['bbox'][1], k['bbox'][2], k['bbox'][3]] # format: [lux, luy, w, h]
                    k['attention_relationship'] = [RELATION_CLASSES.index(r) for r in k['attention_relationship']] # relation index from 0
                    k['spatial_relationship'] = [RELATION_CLASSES.index(r) for r in k['spatial_relationship']]
                    k['contacting_relationship'] = [RELATION_CLASSES.index(r) for r in k['contacting_relationship']]
                    # if max(k['contacting_relationship']) >= 25:
                    #     pdb.set_trace()
                    gt_annotation_frame.append(k)
            gt_annotation_video.append(gt_annotation_frame)

        if len(video) > 0:
            video_list.append(video)
            video_size.append(person_bbox[j]['bbox_size'])
            gt_annotations.append(gt_annotation_video)
        # elif len(video) == 1:
        #     one_frame_video += 1
        # else:
        #     non_person_video += 1

    print('x'*60)
    print('There are {} videos and {} valid frames'.format(len(video_list), valid_nums))
    print('{} videos are invalid (no person), remove them'.format(non_person_video))
    print('{} videos are invalid (only one frame), remove them'.format(one_frame_video))
    print('{} frames have no human bbox in GT, remove them!'.format(non_gt_human_nums))
    
    print('---------- transforming annotations ----------')
    coco_annos = {'videos': [], 'images': [], 'annotations': []}
    coco_annos['categories'] = [{'id': v, 'name': k, 'encode_name': None} for k, v in cat_id_maps.items()] # index from 1
    coco_annos['rel_categories'] = [{'id': v, 'name': k, 'encode_name': None} for k, v in rel_cat_id_maps.items()] # index from 0
    global_fid, global_aid = 1, 1
    for vid, video in tqdm(enumerate(video_list, 1)):
        video_frame_path = os.path.join('frames', video[0].split('/')[0])
        cur_vid_dict = {'id': vid, 'name': video_frame_path, 'vid_train_frames': []}
        for frame, frame_annos in zip(video, gt_annotations[vid-1]):
            frame_idx = int(frame.split('/')[1].split('.')[0])
            if mode == 'train':
                cur_vid_dict['vid_train_frames'].append(frame_idx)
            cur_img_dict = {'file_name': os.path.join('frames', frame), 'id': global_fid, 'frame_id': frame_idx, 'video_id': vid, \
                            'width': video_size[vid-1][0], 'height': video_size[vid-1][1], 'is_vid_train_frame': mode == 'train'}
            coco_annos['images'].append(cur_img_dict)
            for f_anno in frame_annos:
                cur_anno_dict = {'id': global_aid, 'image_id': global_fid, 'video_id': vid, }
                global_aid += 1
                if 'person_bbox' in f_anno.keys():
                    cur_anno_dict['category_id'] = 1
                    cur_anno_dict['bbox'] = f_anno['person_bbox']
                    cur_anno_dict['attention_rel'] = []
                    cur_anno_dict['spatial_rel'] = []
                    cur_anno_dict['contact_rel'] = []
                else:
                    cur_anno_dict['category_id'] = f_anno['class']
                    cur_anno_dict['bbox'] = f_anno['bbox']
                    cur_anno_dict['attention_rel'] = f_anno['attention_relationship']
                    cur_anno_dict['spatial_rel'] = f_anno['spatial_relationship']
                    cur_anno_dict['contact_rel'] = f_anno['contacting_relationship']
                cur_anno_dict.update({'instance_id': -1, 'iscrowd': False, 'occluded': -1, 'generated': -1})
                cur_anno_dict['area'] = cur_anno_dict['bbox'][2] * cur_anno_dict['bbox'][3]
                coco_annos['annotations'].append(cur_anno_dict)                
            global_fid += 1
        coco_annos['videos'].append(cur_vid_dict)
    save_path = os.path.join(root_path, 'annotations', 'ag_{}_coco_style_with_one_frame.json'.format(mode))
    with open(save_path, 'w') as f:
        json.dump(coco_annos, f)

for mode in ['train']: #, 'test']:
    trans_annotations(root_path='../data/action-genome/', mode=mode)