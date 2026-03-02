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
    'classes' : ('Cell-Container', 'Container', 'Dredger', 'Fishing', 'Law Enforce', 'ore-oil','bulk cargo')
},
}
import cv2
import numpy as np
import copy
def poly2obb_le90(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'le90')
    # convert to degrees
    angles = angles * 180 / np.pi
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return (x_ctr.item(), y_ctr.item(), width.item(), height.item(), angles.item())
def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')
def draw_normalized_rotated_boxes(image_path, output_dir, predictions, ground_truths, format='xywha'):#'xywha'
    if type(image_path)!=str:
        image_path=image_path[0]
    image_path='../../InternRS_data/'+image_path
    """
    支持0-1000范围归一化坐标的旋转框可视化
    
    参数说明：
    - image_path: 图片路径
    - output_dir: 输出目录
    - predictions: 预测框列表，支持两种格式：
        • xywha格式: [x_center_norm, y_center_norm, width_norm, height_norm, angle_deg, confidence, class_id]
        • 四边形顶点格式: [[x1_norm,y1_norm], ..., [x4_norm,y4_norm], confidence, class_id]
    - ground_truths: 真值框列表，格式同上（不含置信度）
    - format: 输入格式 ('xywha' 或 'quad')
    """
    
    # 读取图片并获取实际尺寸
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 无法读取图片 {image_path}")
        return
    
    img_height, img_width = img.shape[:2]  # 获取实际图片尺寸

    # 样式配置
    pred_color = (0, 0, 255)    # 红色预测框
    gt_color = (0, 255, 0)      # 绿色真值框
    thickness = 2               # 线宽
    font = cv2.FONT_HERSHEY_SIMPLEX

    def denormalize(point, img_width, img_height):
        """将0-1000范围的坐标转换为实际像素坐标"""
        # print(point)
        x = int(point[0] * img_width / 1000)
        y = int(point[1] * img_height / 1000)
        return (x, y)

    def parse_box(box, is_pred):
        box=copy.deepcopy(box)
        """解析并反归一化旋转框参数"""
        if format == 'xywha':
            # 反归一化中心点和尺寸
            x = box[0] * img_width / 1000
            y = box[1] * img_height / 1000
            w = box[2] * img_width / 1000
            h = box[3] * img_height / 1000
            angle = box[4]
            
            # 生成旋转矩形顶点
            rect = ((x, y), (w, h), angle)
            points = cv2.boxPoints(rect).astype(int)
            
            # 提取元数据
            if is_pred:
                conf, cls = box[5], box[6]
            else:
                cls = box[5]
                conf = None
                
        elif format == 'quad':
            # 反归一化四边形顶点
            points = np.array([denormalize(p, img_width, img_height) for p in box[:4]])
            
            # 提取元数据
            if is_pred:
                conf, cls = box[4], box[5]
            else:
                cls = box[4]
                conf = None
        
        return points, cls, conf

    # 绘制预测框
    for box in predictions:
        points, cls, conf = parse_box(box, is_pred=True)
        
        # 绘制旋转框
        cv2.polylines(img, [points], isClosed=True, color=pred_color, thickness=thickness)
        
        # 添加置信度标签
        label = f"{int(cls)}: {conf:.2f}" if conf else f"{int(cls)}"
        text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
        
        # 标签背景
        txt_pt = tuple(points[0])
        cv2.rectangle(img,
                      (txt_pt[0], txt_pt[1] - text_size[1] - 4),
                      (txt_pt[0] + text_size[0], txt_pt[1]),
                      (40,40,40), -1)
        
        # 标签文字
        cv2.putText(img, label, (txt_pt[0], txt_pt[1] - 4),
                   font, 0.5, (255,255,255), 1)

    # 绘制真值框
    for box in ground_truths:
        points, cls, _ = parse_box(box, is_pred=False)
        
        # 半透明填充
        overlay = img.copy()
        cv2.fillPoly(overlay, [points], gt_color)
        img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
        
        # 绘制边框
        cv2.polylines(img, [points], isClosed=True, color=gt_color, thickness=thickness)
        
        # 添加类别标签
        label = f"GT {int(cls)}"
        cv2.putText(img, label, tuple(points[0]),
                   font, 0.5, gt_color, 1)

    # 保存结果
    output_dir=os.path.join(output_dir,args.output_file.split('/')[-2])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"norm_rot_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, img)
    print(f"可视化结果保存至：{output_path}")
def postprocess_parsed_answer(answers,pred,image_id,cls_map,image_path=None,output_dir='results/vis'):

    gt_labels=[]
    gt_bboxes=[]
    ret=dict()
    if type(answers)==list:
        # answers=answers.split('|||')
        for answer in answers:
            ans_labels = re.findall(r'<ref>(.*?)</ref>', answer)
            ans_boxes = re.findall(r'<box>(.*?)</box>', answer)
            # print(ans_boxes)
            if ans_boxes==[]:
                continue
            ans_labels = [cls_map[ans_label.replace('-',' ').strip()] for ans_label in ans_labels]
            ans_boxes = [ast.literal_eval(ans_box) for ans_box in ans_boxes]
            

            for ans_label,ans_box in zip(ans_labels,ans_boxes):
                for b in ans_box:
                    # print(b)
                    gt_labels.append(ans_label)
                    gt_bboxes.append(b)
                    # print(b)
    else:    
        if answers!='No objects detected.':
            answers=answers.replace('-',' ')
            ans_labels = re.findall(r'<ref>(.*?)</ref>', answers)
            ans_boxes = re.findall(r'<box>(.*?)</box>', answers)
            ans_labels = [cls_map[ans_label] for ans_label in ans_labels]
            ans_boxes = [ast.literal_eval(ans_box) for ans_box in ans_boxes]


            for ans_label,ans_box in zip(ans_labels,ans_boxes):
                for b in ans_box:
                    gt_labels.append(ans_label)
                    gt_bboxes.append(b)#poly2obb_le90()
                    # gt.append(dict(image_id=image_id,class_id=ans_label,bbox=b))
    # else:
    #     return None
    if '|||' in pred:
        preds = pred.split(':')[-1].strip().split('|||')
    else:
        preds = pred.split(':')[-1].strip().split(', ')    
    pred_labels = []
    pred_boxes = []

    for pred in preds:
        # print(pred)
        pred=pred.split('. ')[-1]
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
            # print(pred_box)
            pred_box=ast.literal_eval(pred_box)
            pred_boxes.append(pred_box)
        except SyntaxError:
            print(f"Error: {pred_box} can not be read")
            return None
        except ValueError:
            print(f"Error: {pred_box} can not be convert")
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
            if len(box)!=8:
                print(f'get wrong pred_bbox: {box}')
                continue
            # assert len(box)==5,f'{box}'
            all_bboxes.append(poly2obb_le90(torch.tensor(box,dtype=torch.float32)))
            all_scores.append(1)
            all_labels.append(label)
            # draw_rotated_boxes
            # det.append(dict(image_id=image_id,class_id=label,bbox=box,score=1))
    if all_bboxes==[]:
        print('nono')
        # return None
        # all_bboxes=[[]]
    # print(gt_bboxes)
    gt_instances=dict(labels=torch.as_tensor(gt_labels,dtype=int),bboxes=torch.empty(1,5) if gt_bboxes==[] else torch.as_tensor(gt_bboxes,dtype=int))
    pred_instances=dict(labels=torch.as_tensor(all_labels,dtype=int),bboxes=torch.empty(1,5) if all_bboxes==[] else torch.as_tensor(all_bboxes,dtype=int),scores=torch.as_tensor(all_scores,dtype=int))
    ignored_instances=dict(labels=torch.as_tensor([0]),bboxes=torch.as_tensor([[0,0,0,0,0]]))
    ret=dict(img_id=image_id,gt_instances=gt_instances,pred_instances=pred_instances,ignored_instances=ignored_instances)
    if image_path!=None:
        predictions=[]
        ground_truths=[]
        for all_bbox,all_score,all_label in zip(all_bboxes,all_scores,all_labels):
            prediction=[]
            prediction.extend(all_bbox)
            prediction.append(all_score)
            prediction.append(all_label)
            predictions.append(prediction)
        for gt_bbox,gt_label in zip(gt_bboxes,gt_labels):
            ground_truth=[]
            ground_truth.extend(gt_bbox)
            ground_truth.append(gt_label)
            ground_truths.append(ground_truth)
        # print(ground_truths)
        draw_normalized_rotated_boxes(image_path, output_dir, predictions, ground_truths, format='xywha')
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
            ('Cell-Container', 'Container', 'Dredger', 'Fishing', 'Law Enforce', 'ore-oil','bulk cargo'),
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
        answers=data['gt_answers']
        if type(answers)!=list:
            # answers='|||'.join(answers)
        # else:
            answers=answers['value']
        if i%100==0:
            ret=postprocess_parsed_answer(answers,data['answer'],i,cls_map,data['image_name'])#[0])
        else:
            ret=postprocess_parsed_answer(answers,data['answer'],i,cls_map)
        
        if ret==None:
            continue
        # print(ret['gt_instances']['bboxes'].shape)
        results.append(ret)
        # break
        
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