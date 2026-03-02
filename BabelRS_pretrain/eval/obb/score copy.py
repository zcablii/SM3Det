import argparse
import json
import re
import ast
import torch
import os
import numpy as np
import cv2
from collections import defaultdict
import cv2
from functools import cmp_to_key
from mmengine.evaluator import DumpResults, Evaluator
from mmengine import dump
from rapidfuzz import process
ds_collections = {
'DOTA': {
    'classes':('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank', 'soccer-ball-field',
                'roundabout', 'harbor', 'swimming-pool', 'helicopter')
},
'FAIR1M2': {
    'classes' : ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
                'A321', 'A330', 'A350', 'ARJ21', 'Passenger Ship', 'Motorboat',
                'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship',
                'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
                'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator',
                'Truck Tractor', 'Basketball Court', 'Tennis Court', 'Football Field',
                'Baseball Field', 'Intersection', 'Roundabout', 'Bridge')
},
'RSAR': {
    'classes' : ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
},
'SRSDD': {
    'classes' : ('Cell-Container', 'Container', 'Dredger', 'Fishing', 'LawEnforce', 'ore-oil')
},
}
def postprocess_parsed_answer(answer,pred,image_id,cls_map):

    gt_labels=[]
    gt_bboxes=[]
    ret=dict()
    if answer!='No objects detected.':
        answer=answer.replace('-',' ')
        ans_labels = re.findall(r'<ref>(.*?)</ref>', answer)
        ans_boxes = re.findall(r'<box>(.*?)</box>', answer)
        ans_labels = [cls_map[ans_label] for ans_label in ans_labels]
        ans_boxes = [ast.literal_eval(ans_box) for ans_box in ans_boxes]


        for ans_label,ans_box in zip(ans_labels,ans_boxes):
            for b in ans_box:
                gt_labels.append(ans_label)
                gt_bboxes.append(b)
                # gt.append(dict(image_id=image_id,class_id=ans_label,bbox=b))
    # else:
    #     return None

    preds = pred.split(':')[-1].strip().split(', ')
    pred_labels = []
    pred_boxes = []

    for pred in preds:
        # print(pred)
        pred_label=re.findall(r'(.*?)\[',pred)
        if pred_label==[]:
            continue
        else:
            # print(pred_label)
            pred_label=pred_label[0]
        # print(pred_label)
        pred_labels.append(pred_label)
        
        pred_box=pred.replace(pred_label,'')
        if pred_box[-1]!=']':
            return None
            pred_box=pred_box[:pred_box.rfind(']')+1]+']'
        try:
            pred_box=ast.literal_eval(pred_box)
            pred_boxes.append(pred_box)
        except SyntaxError:
            print(f"Error:{pred_box} can not be read")
            return None
    all_bboxes=[]
    all_labels=[]
    all_scores=[]
    det=[]
    for pred_label,pred_box in zip(pred_labels,pred_boxes):
        if pred_label.strip() == "":
            continue
        if pred_label in cls_map:
            label = cls_map[pred_label.lower()]
        else:
            fuzzy_matched_cat = process.extractOne(pred_label.lower(), cls_map.keys())[0]
            # if logger is None:
            print(f"Fuzzy matched {pred_label.lower()} to {fuzzy_matched_cat}")
            # else:
                # logger.info(f"Fuzzy matched {pred_label.lower()} to {fuzzy_matched_cat}")
            label = cls_map[fuzzy_matched_cat]
        for box in pred_box:
            if len(box)!=5:
                print(f'get wrong pred_bbox: {box}')
                continue
            # assert len(box)==5,f'{box}'
            all_bboxes.append(box)
            all_scores.append(1)
            all_labels.append(label)

            # det.append(dict(image_id=image_id,class_id=label,bbox=box,score=1))
    if all_bboxes==[]:
        print('noobject')
        # return None
        # all_bboxes=[[]]
    gt_instances=dict(labels=torch.as_tensor(gt_labels,dtype=int),bboxes=torch.empty(1,5) if gt_bboxes==[] else torch.as_tensor(gt_bboxes,dtype=int))
    pred_instances=dict(labels=torch.as_tensor(all_labels,dtype=int),bboxes=torch.empty(1,5) if all_bboxes==[] else torch.as_tensor(all_bboxes,dtype=int),scores=torch.as_tensor(all_scores,dtype=int))
    ignored_instances=dict(labels=torch.as_tensor([0]),bboxes=torch.as_tensor([[0,0,0,0,0]]))
    ret=dict(img_id=image_id,gt_instances=gt_instances,pred_instances=pred_instances,ignored_instances=ignored_instances)
    return ret
# return gt,det
def prepare_evaluator(dataset_name):
    if dataset_name == "DOTA":
        from mmrotate.evaluation import DOTAMetric
        evaluator = Evaluator(DOTAMetric(metric="mAP"))
        evaluator.dataset_meta = {
            'classes':
            ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
                'harbor', 'swimming-pool', 'helicopter'),
        }
    elif dataset_name == "FAIR1M2":
        from mmrotate.evaluation import DOTAMetric
        # from lmmrotate.modules.fair_metric import FAIRMetric
        evaluator = Evaluator(DOTAMetric(metric="mAP"))
        evaluator.dataset_meta = {
            'classes':
            ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
            'A321', 'A330', 'A350', 'ARJ21', 'Passenger Ship', 'Motorboat',
            'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship',
            'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
            'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator',
            'Truck Tractor', 'Basketball Court', 'Tennis Court', 'Football Field',
            'Baseball Field', 'Intersection', 'Roundabout', 'Bridge'),
        }
    elif dataset_name == "dior":
        from mmrotate.evaluation import DOTAMetric
        evaluator = Evaluator(DOTAMetric(metric="mAP"))
        evaluator.dataset_meta = {
            'classes':
            ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
                'chimney', 'expressway-service-area', 'expressway-toll-station',
                'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
                'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'),
        }
    elif dataset_name == "SRSDD":
        from mmrotate.evaluation import DOTAMetric
        # from mmrotate.evaluation import RotatedCocoMetric
        evaluator = Evaluator(DOTAMetric(metric="mAP"))#RotatedCocoMetric(metric='bbox', classwise=True))
        evaluator.dataset_meta = {
            'classes':
            ('Cell-Container', 'Container', 'Dredger', 'Fishing', 'LawEnforce', 'ore-oil'),
        }
    elif dataset_name == "RSAR":
        from mmrotate.evaluation import DOTAMetric
        # from mmrotate.evaluation import RotatedCocoMetric
        evaluator = Evaluator(DOTAMetric(metric="mAP"))#RotatedCocoMetric(metric='bbox', classwise=True))
        evaluator.dataset_meta = {
            'classes' : ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
        }
    return evaluator

def calculate_scores(datas,evaluator,cls_map):
    ground_truths=[]
    detections=[]
    results=[]
    for i,data in enumerate(datas):
        ret=postprocess_parsed_answer(data['gt_answers']['value'],data['answer'],i,cls_map)
        
        if ret==None:
            continue
        # print(ret['gt_instances']['bboxes'].shape)
        results.append(ret)
        
        # ground_truths.extend(gt)
        # detections.extend(det)
    mAP = evaluator.offline_evaluate(data_samples=results, chunk_size=128)
    pickle_results_path = f"{os.path.dirname(args.output_file)}/{args.dataset}/output.pkl"
    dump(results, pickle_results_path)
    print(mAP)
    return mAP
    # print(mAP)
    # calculate_mAP(ground_truths, detections)
def cal_f1():
    #fixed
    prediction_path=f"{os.path.dirname(args.output_file)}/{args.dataset}/output.pkl"
    cmd = f'python eval/obb/f1_metric.py {prediction_path} --output_file {args.output_file}'
    #cmd = f'python eval/rs_det/caculate.py --output_file {output_path}'
    print(cmd)
    os.system(cmd)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DOTA')
    parser.add_argument('--output_file', type=str, default='')
    # parser.add_argument('--results_path', type=str, default='')
    args = parser.parse_args()

    with open(args.output_file, 'r') as f:
        data = json.load(f)
    if 'outputs' in data:
        data = data['outputs']
    cls_map = {c.replace("-", " ").lower(): i
        for i, c in enumerate(ds_collections[args.dataset]['classes'])
    }
    print(cls_map)
    evaluator = prepare_evaluator(args.dataset)
    mAP=calculate_scores(data,evaluator,cls_map)

    results = {
        # 'type_scores': type_scores,
        # 'type_counts': type_counts,
        # 'total_score': total_score,
        'mAP' : mAP,
        'outputs': data
    }
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    cal_f1()