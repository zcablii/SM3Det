# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmengine.fileio import load
from mmengine.utils import ProgressBar
from mmengine.evaluator import DumpResults

from mmrotate.structures.bbox import qbox2rbox, rbbox_overlaps
from mmrotate.utils import register_all_modules
import json

# from lmmrotate.utils import monkey_patch_of_collections_typehint_for_mmrotate1x
import collections
from collections.abc import Mapping, Sequence, Iterable
collections.Mapping = Mapping
collections.Sequence = Sequence
collections.Iterable = Iterable

# monkey_patch_of_collections_typehint_for_mmrotate1x()


def get_num_classes(results):
    min_label_id = 0
    max_label_id = -1
    for per_img_res in results:
        gt_labels = per_img_res['gt_instances']['labels']
        if len(gt_labels) > 0:
            min_label_id = min(min_label_id, gt_labels.min().item())
            max_label_id = max(max_label_id, gt_labels.max().item())
    return max_label_id - min_label_id + 1


def calculate_confusion_matrix(results,
                               score_thr=0,
                               nms_iou_thr=None,
                               tp_iou_thr=0.5):
    num_classes = get_num_classes(results)
    confusion_matrix = torch.zeros(size=[num_classes + 1, num_classes + 1])
    prog_bar = ProgressBar(len(results))
    for per_img_res in results:
        pred_instances = per_img_res['pred_instances']
        gt_instances = per_img_res['gt_instances']
        confusion_matrix = analyze_per_img_dets(
            confusion_matrix, gt_instances, pred_instances, 
            score_thr, tp_iou_thr, nms_iou_thr
        )
        prog_bar.update()
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix,
                         gt_instances,
                         pred_instances,
                         score_thr=0,
                         tp_iou_thr=0.5,
                         nms_iou_thr=None):
    gt_bboxes = gt_instances['bboxes']
    gt_labels = gt_instances['labels']
    # print(gt_bboxes)
    if gt_bboxes.shape[1] == 8:
        gt_bboxes = qbox2rbox(gt_bboxes)

    unique_label = torch.unique(pred_instances['labels'])
    true_positives = torch.zeros(len(gt_labels))
    for det_label in unique_label:
        mask = (pred_instances['labels'] == det_label)
        det_bboxes = pred_instances['bboxes'][mask]
        det_scores = pred_instances['scores'][mask]

        if nms_iou_thr:
            det_bboxes, _ = nms_rotated(det_bboxes, det_scores, nms_iou_thr)
        ious = rbbox_overlaps(det_bboxes[:, :5].float(), gt_bboxes.float())
        for i, score in enumerate(det_scores):
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:  # BG FP
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN
            confusion_matrix[gt_label, -1] += 1
    return confusion_matrix


def confusion_matrix_to_ap_ar_f1(confusion_matrix):
    TP = torch.diag(confusion_matrix)
    FP = torch.sum(confusion_matrix, axis=0) - TP
    FN = torch.sum(confusion_matrix, axis=1) - TP

    def _safe(arr):
        arr[arr == 0] = 1
        return arr

    precision = TP / _safe(TP + FP)
    recall = TP / _safe(TP + FN)
    average_precision = torch.mean(precision)
    average_recall = torch.mean(recall)
    f1 = 2 * (average_precision * average_recall) / (average_precision + average_recall)
    return average_precision.item(), average_recall.item(), f1.item()


def main():
    register_all_modules()

    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_path', help='Path to prediction file')
    parser.add_argument('--output_file', help='Path to prediction file')
    parser.add_argument('--score_thr', type=float, default=None, nargs='+')
    parser.add_argument('--nms_iou_thr', type=float, default=None, nargs='+')
    parser.add_argument('--tp_iou_thr', type=float, default=None, nargs='+')
    args = parser.parse_args()

    results = load(args.prediction_path)

    if args.score_thr is None:
        args.score_thr = [0.9]#[x * 0.1 for x in range(10)]
    if args.nms_iou_thr is None:
        args.nms_iou_thr = [None]
    if args.tp_iou_thr is None:
        args.tp_iou_thr = [0.5]

    all_average_precision, all_average_recall, all_f1 = [], [], []
    for nms_iou_thr in args.nms_iou_thr:
        for score_thr in args.score_thr:
            for tp_iou_thr in args.tp_iou_thr:
                print(f"\n\nscore_thr: {score_thr}, nms_iou_thr: {nms_iou_thr}, tp_iou_thr: {tp_iou_thr}")
                confusion_matrix = calculate_confusion_matrix(results, score_thr, nms_iou_thr, tp_iou_thr)
                average_precision, average_recall, f1 = confusion_matrix_to_ap_ar_f1(confusion_matrix)

                print("-"*5, f"\n\nscore_thr: {score_thr}, nms_iou_thr: {nms_iou_thr}, tp_iou_thr: {tp_iou_thr}", "-"*5)
                print("\n\nAP:  ", average_precision)
                print("AR: ", average_recall)
                print("F1: ", f1)
                print("-"*5, f"\n\nscore_thr: {score_thr}, nms_iou_thr: {nms_iou_thr}, tp_iou_thr: {tp_iou_thr}", "-"*5)

                all_average_precision.append(average_precision)
                all_average_recall.append(average_recall)
                all_f1.append(f1)

    print("Mean F1: ", sum(all_f1) / len(all_f1))
    print("Max F1: ", max(all_f1))
    print("Min F1: ", min(all_f1))

    with open(args.output_file, 'r') as f:
        data = json.load(f)
    if 'outputs' in data:
        results = {
            'mAP' : data['mAP'],
            'mF1' : sum(all_f1) / len(all_f1),
            'outputs': data['outputs']
        }
    else:
        results ={
            'mF1' : sum(all_f1) / len(all_f1),
            'outputs': data
        }
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()