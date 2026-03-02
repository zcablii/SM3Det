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
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
def cmp(a, b, c):
    if a.x >= 0 and b.x < 0:
        return -1
    if a.x == 0 and b.x == 0:
        # return a.y > b.y
        if a.y > b.y:
            return -1
        elif a.y < b.y:
            return 1
        return 0
    det = (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y)
    if det < 0:
        return 1
    if det > 0:
        return -1
    d1 = (a.x - c.x) * (a.x - c.x) + (a.y - c.y) * (a.y - c.y)
    d2 = (b.x - c.x) * (b.x - c.x) + (b.y - c.y) * (b.y - c.y)
    # return d1 > d2
    if d1 > d2:
        return -1
    elif d1 < d2:
        return 1
    return 0
def rotated_iou(box1, box2):
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
    r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        intersec = cv2.contourArea(order_pts)
        union = area1 + area2 - intersec
        iou = intersec * 1.0 / union
    else:
        iou = 0.0
    return iou

def compute_ap(recall, precision):
    """计算平均精度（AP）使用VOC11点插值法"""
    # 扩展recall和precision
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # 确保precision单调递减
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = np.maximum(mpre[i], mpre[i+1])
    
    # 找到recall变化的点
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # 计算AP
    ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap

def calculate_mAP(ground_truths, detections, iou_threshold=0.5):
    """
    计算旋转框的mAP
    :param ground_truths: 真实框列表，每个元素包含：
        {'image_id': ..., 'class_id': ..., 'rotated_box': [x,y,w,h,angle]}
    :param detections: 检测框列表，每个元素包含：
        {'image_id': ..., 'class_id': ..., 'rotated_box': [x,y,w,h,angle], 'confidence': ...}
    :param iou_threshold: IoU阈值
    :return: mAP值
    """
    # 按类别分组真实框
    gt_dict = defaultdict(lambda: defaultdict(list))
    for gt in ground_truths:
        gt_dict[gt['class_id']][gt['image_id']].append(gt['bbox'])
    
    # 按类别分组检测框
    det_dict = defaultdict(list)
    for det in detections:
        det_dict[det['class_id']].append(det)
    
    aps = []
    
    for class_id in det_dict:
        # 获取当前类别的所有检测并按置信度排序
        class_dets = sorted(det_dict[class_id], key=lambda x: x['score'], reverse=True)
        class_gts = gt_dict.get(class_id, {})
        
        # 统计总真实框数
        total_gts = sum(len(gts) for gts in class_gts.values())
        if total_gts == 0:
            continue
        
        # 初始化变量
        tp = np.zeros(len(class_dets))
        fp = np.zeros(len(class_dets))
        used_gts = defaultdict(dict)  # {image_id: {index: bool}}
        
        # 初始化每个图像的已使用标记
        for image_id in class_gts:
            used_gts[image_id] = [False] * len(class_gts[image_id])
        
        # 处理每个检测
        for det_idx, det in enumerate(class_dets):
            image_id = det['image_id']
            det_box = det['bbox']
            
            if image_id not in class_gts:
                fp[det_idx] = 1
                continue
                
            # 获取该图像的所有真实框
            gts = class_gts[image_id]
            best_iou = 0.0
            best_gt_idx = -1
            
            # 寻找最佳匹配的真实框
            for gt_idx, gt_box in enumerate(gts):
                if not used_gts[image_id][gt_idx]:
                    iou = rotated_iou(det_box, gt_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # 判断是否匹配成功
            if best_gt_idx != -1:
                tp[det_idx] = 1
                used_gts[image_id][best_gt_idx] = True
            else:
                fp[det_idx] = 1
        
        # 计算累计统计
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算召回率和精确率
        recall = tp_cumsum / total_gts
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # 计算AP
        ap = compute_ap(recall, precision)
        aps.append(ap)
    
    # 计算mAP
    mAP = np.mean(aps) if aps else 0.0
    print(mAP)
    return mAP
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
    
    if answer=='No objects detected':
        return None,None
    ret=dict()
    
    ans_labels = re.findall(r'<ref>(.*?)</ref>', answer)
    ans_boxes = re.findall(r'<box>(.*?)</box>', answer)
    ans_labels = [cls_map[ans_label] for ans_label in ans_labels]
    ans_boxes = [ast.literal_eval(ans_box) for ans_box in ans_boxes]
    
    gt=[]
    for ans_label,ans_box in zip(ans_labels,ans_boxes):
        for b in ans_box:
            gt.append(dict(image_id=image_id,class_id=ans_label,bbox=b))

    
    preds = pred.split(':')[-1].strip().split(', ')
    pred_labels = []
    pred_boxes = []
    
    for pred in preds:
        # print(pred)
        pred_label=re.findall(r'(.*?)\[',pred)[0]
        # print(pred_label)
        pred_labels.append(pred_label)
        
        pred_box=pred.replace(pred_label,'')
        if pred_box[-1]!=']':
            pred_box=pred_box[:pred_box.rfind(']')+1]+']'
        pred_box=ast.literal_eval(pred_box)
        pred_boxes.append(pred_box)

    all_boxes=[]
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
            if logger is None:
                print(f"Fuzzy matched {ref.lower()} to {fuzzy_matched_cat}")
            else:
                logger.info(f"Fuzzy matched {pred_label.lower()} to {fuzzy_matched_cat}")
            label = cls_map[fuzzy_matched_cat]
        for box in pred_box:
            assert len(box)==5
            det.append(dict(image_id=image_id,class_id=label,bbox=box,score=1))

    return gt,det
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
    elif dataset_name == "fair":
        from lmmrotate.modules.fair_metric import FAIRMetric
        evaluator = Evaluator(FAIRMetric(metric="mAP"))
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
    elif dataset_name == "srsdd":
        from mmrotate.evaluation import RotatedCocoMetric
        evaluator = Evaluator(RotatedCocoMetric(metric='bbox', classwise=True))
        evaluator.dataset_meta = {
            'classes':
            ('Cell-Container', 'Container', 'Dredger', 'Fishing', 'LawEnforce', 'ore-oil'),
        }
    return evaluator

def calculate_scores(datas,cls_map):
    ground_truths=[]
    detections=[]
    for i,data in enumerate(datas):
        gt,det=postprocess_parsed_answer(data['gt_answers']['value'],data['answer'],i,cls_map)
        if gt==None:
            continue
        ground_truths.extend(gt)
        detections.extend(det)

    calculate_mAP(ground_truths, detections)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DOTA')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--results_path', type=str, default='')
    args = parser.parse_args()

    with open(args.output_file, 'r') as f:
        data = json.load(f)
    if 'outputs' in data:
        data = data['outputs']
    cls_map = {c.replace("-", " ").lower(): i
        for i, c in enumerate(ds_collections[args.dataset]['classes'])
    }
    calculate_scores(data,cls_map)

    results = {
        # 'type_scores': type_scores,
        # 'type_counts': type_counts,
        # 'total_score': total_score,
        # 'total_score_useful': total_score_useful,
        'outputs': data
    }
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
