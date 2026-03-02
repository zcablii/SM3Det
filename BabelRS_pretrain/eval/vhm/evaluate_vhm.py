import os
import re
import math
import itertools
import requests
import json
import argparse
from io import BytesIO
from collections import defaultdict
import random
import time
from functools import partial
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
# import pandas as pd
# import shortuuid
from collections import Counter
# from vhm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from vhm.conversation import conv_templates, SeparatorStyle
# from vhm.model.builder import load_pretrained_model
# from vhm.utils import disable_torch_init
# from vhm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from vhm_model import VHM
# from smp import dump
import jsonlines

ds_collections = {
    'cls_AID': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_AID.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'cls',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'cls_AID_100': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_AID_100.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'cls',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'cls_AID_1': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_AID_1.jsonl',
        'max_new_tokens': 50,
        'min_new_tokens': 1,
        'type': 'cls',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'cls_METER_ML': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_METER_ML.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'cls_NWPU_RESISC45': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_NWPU_RESISC45.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'cls_SIRI_WHU': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_SIRI_WHU.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'cls_WHU_RS19': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_WHU_RS19.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'cls_WHU_RS19_hard': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_WHU_RS19_hard.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'cls_millionAID': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_rgb_Million-AID_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'cls_sar_TenGeoP':{
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_sar_TenGeoP-SARwv_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'cls_sar_ISPRS':{
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_sar_ISPRS_SAR_classification_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'cls_fmow':{
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_rgb_fmow_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },

    'cls_UCM':{
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_rgb_UCM_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    
    'cls_sar_Sentinel':{
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_sar_Sentinel_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
    'mcq_test':{
        'root': '../../InternRS_data/val_annos/VHM_eval/mcq_merged_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval'
    },
}

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def convt_qa(conversations, task_type):
    values = [conversation['value'] for conversation in conversations]
    query = values[0]
    answer = values[1]

    if 'cls' in task_type:
        query = '{CLS}' +' '+ query
    elif 'vqa' in task_type:
        query = '{VQA}' +' '+ query 
    elif 'vg' in task_type:
        query = '{VG}' +' '+ query
    elif 'color' in task_type or 'presence' in task_type or 'position' in task_type:
        query = '{IDK}' +' '+ query
    else:
        query = '{IT}' +' '+ query

    return query, answer


def eval_results_presence(args, result_json_file, save_excel):
    # data['yes'] = data["prediction"].str.contains("Yes", case=False)
    # data["no"] = data["prediction"].str.contains("No", case=False) | data["prediction"].str.contains("False", case=False)
    # # data["no"] = data["prediction"].str.contains("No", case=False)
    # data['raw_prediction'] = data['prediction']
    # data['prediction'] = data.apply(
    #     lambda x: "Yes" if x["yes"] and not x["no"] else "No" if x["no"] and not x["yes"] else "Unknown", axis=1
    # )
    # data.drop(["yes", "no"], axis=1, inplace=True)
    # data["score"] = (data["answer"].str.lower() == data["prediction"].str.lower())
    with open(result_json_file, 'r') as f:
        result_lines = f.readlines()

    dataset_size = len(result_lines)
    ret = {}
    final_dict = defaultdict(list)

    # score = 0
    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        raw_prediction = result_dict['pred']
        answer = result_dict['answer']

        x_yes = True if 'yes' in raw_prediction.lower() else False
        x_no = True if any([w in raw_prediction for w in ('no', 'false')]) else False
        prediction = "yes" if x_yes and not x_no else "no" if x_no and not x_yes else "unknown"

        final_dict['score'].append(answer.lower()==prediction.lower())

        final_dict['filename'].append(str(os.path.basename(result_dict['filename'])))
        final_dict['size'].append(str(result_dict['size']))
        final_dict['query'].append(str(result_dict['query']))
        final_dict['answer'].append(str(result_dict['answer']))
        final_dict['prediction'].append(str(prediction))
        final_dict['raw_prediction'].append(str(result_dict['pred']))

    avg_score = sum(final_dict['score']) / len(final_dict['score'])
    perf_dict = {
        'perception': avg_score,
    }
    ret.update(perf_dict)
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    dump(ret, save_excel)

    print(f'perception: {avg_score}')
    return perf_dict


def eval_results_color(args, result_json_file, save_excel):
    # elif 'color' in dataname:
    # data['raw_prediction'] = data['prediction'].astype(str)
    # data['prediction'] = data['raw_prediction'].apply(
    #     lambda s: s.replace(" ", "").lower()
    # )
    # data["answer"] = data["answer"].astype(str)
    # data['answer'] = data['answer'].apply(
    #     lambda s: s.replace(" ", "").lower().split(",")
    # )
    # data["score"] = data.apply(
    # lambda row: sum(ans in row["prediction"] for ans in row["answer"])/len(row["answer"]), axis=1
    # )
    with open(result_json_file, 'r') as f:
        result_lines = f.readlines()

    dataset_size = len(result_lines)
    ret = {}
    final_dict = defaultdict(list)

    # score = 0
    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        raw_prediction = result_dict['pred']
        answer = result_dict['answer']

        prediction = raw_prediction.replace(" ", "").lower()
        answer = answer.replace(" ", "").lower().split(",")

        final_dict['score'].append(sum(ans in prediction for ans in answer)/len(answer))

        final_dict['filename'].append(str(os.path.basename(result_dict['filename'])))
        final_dict['size'].append(str(result_dict['size']))
        final_dict['query'].append(str(result_dict['query']))
        final_dict['answer'].append(str(result_dict['answer']))
        final_dict['prediction'].append(str(prediction))
        final_dict['raw_prediction'].append(str(result_dict['pred']))

    avg_score = sum(final_dict['score']) / len(final_dict['score'])
    perf_dict = {
        'perception': avg_score,
    }
    ret.update(perf_dict)
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    dump(ret, save_excel)

    print(f'perception: {avg_score}')
    return perf_dict


def eval_results_cls_vqa(args, result_json_file, save_excel):

    with open(result_json_file, 'r') as f:
        result_lines = f.readlines()

    dataset_size = len(result_lines)
    ret = {}
    final_dict = defaultdict(list)

    # score = 0
    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        raw_prediction = str(result_dict['pred'])
        answer = str(result_dict['answer'])

        prediction = raw_prediction.replace(" ", "").lower()
        answer = answer.replace(" ", "").lower()

        final_dict['score'].append(prediction in answer)

        final_dict['filename'].append(str(os.path.basename(result_dict['filename'])))
        final_dict['size'].append(str(result_dict['size']))
        final_dict['query'].append(str(result_dict['query']))
        final_dict['answer'].append(str(result_dict['answer']))
        final_dict['prediction'].append(str(prediction))
        final_dict['raw_prediction'].append(str(result_dict['pred']))

    avg_score = sum(final_dict['score']) / len(final_dict['score'])
    perf_dict = {
        'perception': avg_score,
    }
    ret.update(perf_dict)
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    dump(ret, save_excel)

    print(f'perception: {avg_score}')
    return perf_dict


def eval_results_multichoice(args, result_json_file, save_excel):

    with open(result_json_file, 'r') as f:
        result_lines = f.readlines()

    dataset_size = len(result_lines)
    ret = {}
    final_dict = defaultdict(list)

    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        raw_prediction = result_dict['pred']
        answer = result_dict['answer']

        # NOTE: this will cause wrong matching
        # prediction = ''.join(sorted(set(raw_prediction) & set("ABCDE"))) or None
        prediction = next((char for char in raw_prediction if char in 'ABCDE'), None)

        final_dict['score'].append(prediction.lower() in answer.lower())

        final_dict['filename'].append(str(os.path.basename(result_dict['filename'])))
        final_dict['size'].append(str(result_dict['size']))
        final_dict['query'].append(str(result_dict['query']))
        final_dict['answer'].append(str(result_dict['answer']))
        final_dict['prediction'].append(str(prediction))
        final_dict['raw_prediction'].append(str(result_dict['pred']))

    avg_score = sum(final_dict['score']) / len(final_dict['score'])
    perf_dict = {
        'perception': avg_score,
    }
    ret.update(perf_dict)
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    dump(ret, save_excel)

    print(f'perception: {avg_score}')
    return perf_dict


def eval_results_couting(args, result_json_file, save_excel):
    def convert_to_words_100(num):
        def one_to_nineteen(n):
            words = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
                    "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", 
                    "Eighteen", "Nineteen"]
            return words[n-1].lower()

        def tens(n):
            words = ["Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
            return words[n-2].lower()

        if num == 0:
            return "zero"
        if num < 20:
            return one_to_nineteen(num)
        if num < 100:
            if num % 10 == 0:
                return tens(num // 10)
            else:
                return tens(num // 10) + "-" + one_to_nineteen(num % 10)
    
    def word_to_num_100(sentence):
        # Mapping of number words to their numeral equivalents

        number_words = {convert_to_words_100(i): i for i in range(100)}

        # Split the sentence into words and look for number words
        for word in sentence.split():
            if word.lower() in number_words:
                return str(number_words[word.lower()])
        return None

    def parse_number(x):
        if re.search(r'\d+', x):
            return re.search(r'\d+', x).group()
        else:
            extract_number = word_to_num_100(x)
            return extract_number if extract_number is not None else ''


    COUNTING_LEVEL = (2, 6, 12, float('inf'))
    LEVEL_NAME = ('low', 'medium', 'high', 'ultra')

    with open(result_json_file, 'r') as f:
        result_lines = f.readlines()

    dataset_size = len(result_lines)
    ret = {}
    final_dict = defaultdict(list)

    level_count = np.zeros(len(COUNTING_LEVEL)+1)
    level_score = np.zeros(len(COUNTING_LEVEL)+1)
    level_parsing_failure = np.zeros(len(COUNTING_LEVEL)+1)
    global_MAE_list = []
    global_MAPE_list = []
    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        raw_prediction = result_dict['pred']
        answer = int(result_dict['answer'])
        prediction = int(parse_number(raw_prediction))
        score = answer==prediction
        l = 0
        while answer > COUNTING_LEVEL[l]:
            l += 1
        level_count[l] += 1
        level_score[l] += score
        # if str(prediction).lower() == 'nan':
        if prediction is None or prediction == '':
            level_parsing_failure[l] += 1
        else:
        # ====== calculate MAE and MAPE ======
        # if str(prediction).lower() != 'nan':
            global_MAE_list.append(abs(int(prediction)-int(answer)))
            global_MAPE_list.append(abs((int(prediction)-int(answer)) / int(answer)))

        final_dict['filename'].append(str(os.path.basename(result_dict['filename'])))
        final_dict['size'].append(str(result_dict['size']))
        final_dict['query'].append(str(result_dict['query']))
        final_dict['answer'].append(str(result_dict['answer']))
        final_dict['prediction'].append(str(prediction))
        final_dict['raw_prediction'].append(str(result_dict['pred']))

    counting_Ret = defaultdict(list)
    # ====== calculate MAE and MAPE ======
    counting_Ret['MAE'] = sum(global_MAE_list) / len(global_MAE_list)
    counting_Ret['MAPE'] = sum(global_MAPE_list) / len(global_MAPE_list)

    last_level = 0
    for l in range(len(COUNTING_LEVEL)):
        acc_value = level_score[l] / level_count[l] * 100.0 if level_count[l] > 0 else -1  # float('nan')
        counting_Ret['COUNTING_RANGE'].append(f'({last_level}, {COUNTING_LEVEL[l]}]')
        counting_Ret['RANGE_COUNT'].append(level_count[l])
        counting_Ret['RANGE_ACC'].append(acc_value)
        counting_Ret['PARSING_FAILURE_COUNT'].append(level_parsing_failure[l])
        counting_Ret['PARSING_FAILURE_RATIO'].append(level_parsing_failure[l] / dataset_size)
        last_level = COUNTING_LEVEL[l]


    perf_dict = {
        'MAE': counting_Ret["MAE"],
        'MAPE': counting_Ret["MAPE"],
    }
    ret.update(counting_Ret)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    dump(ret, save_excel)

    print(f'MAE: {counting_Ret["MAE"]}')
    print(f'MAPE: {counting_Ret["MAPE"]}')
    return perf_dict


EXTRACT_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")
def eval_results_bbox(args, result_json_file, save_excel, iou_threshold=0.5):

    def extract_bbox(text):
        start_index = text.find('[')
        end_index = text.rfind(']')
        if start_index != -1 and end_index != -1:
            answer_numbers = EXTRACT_NUMBER_PATTERN.findall(text[start_index:end_index+1])
            return [float(number) for number in answer_numbers]
        else:
            return None

    def intersection_geo(box1, box2):
        # 解包两个矩形框的坐标
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2

        # 计算交集的坐标
        x_min_int = max(x_min1, x_min2)
        y_min_int = max(y_min1, y_min2)
        x_max_int = min(x_max1, x_max2)
        y_max_int = min(y_max1, y_max2)

        return x_min_int, y_min_int, x_max_int, y_max_int

    def calculate_area(box):
        x_min1, y_min1, x_max1, y_max1 = box
        area_box1 = (x_max1 - x_min1) * (y_max1 - y_min1)
        return area_box1

    def calculate_iou(box1, box2):
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2
        x_min_int, y_min_int, x_max_int, y_max_int = intersection_geo(box1, box2)

        # 如果没有交集，直接返回0
        if x_min_int >= x_max_int or y_min_int >= y_max_int:
            return 0.0

        # 计算交集的面积
        area_int = (x_max_int - x_min_int) * (y_max_int - y_min_int)

        area_box1 = (x_max1 - x_min1) * (y_max1 - y_min1)
        area_box2 = (x_max2 - x_min2) * (y_max2 - y_min2)
        iou = area_int / (area_box1 + area_box2 - area_int)
        return iou


    with open(result_json_file, 'r') as f:
        result_lines = f.readlines()
    
    dataset_size = len(result_lines)
    AREA_LEVEL = (32**2, 96**2, float('inf'))
    LEVEL_NAME = ('S', 'M', 'L')
    level_count = np.zeros(len(AREA_LEVEL))
    level_hit_count = np.zeros(len(AREA_LEVEL))

    ret = {}
    final_dict = defaultdict(list)
    hit_count = 0
    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        h, w = result_dict['size']
        answer_bbox = extract_bbox(str(result_dict['answer']))
        pred_bbox_ori = extract_bbox(result_dict['pred'])

        l = 0
        while calculate_area(answer_bbox) > AREA_LEVEL[l]:
            l += 1
        level_count[l] += 1

        if answer_bbox and pred_bbox_ori:
            pred_bbox = [
                float(pred_bbox_ori[0] * w / 1000.0),
                float(pred_bbox_ori[1] * h / 1000.0),
                float(pred_bbox_ori[2] * w / 1000.0),
                float(pred_bbox_ori[3] * h / 1000.0),
            ]
            iou = calculate_iou(answer_bbox, pred_bbox)
            print(f'[{img_i}/{len(result_lines)}] answer_bbox:{answer_bbox}, pred_bbox:{pred_bbox}, iou:{iou}')

            if iou >= iou_threshold:
                # hit_count += 1
                level_hit_count[l] += 1
        else:
            pred_bbox = None
        
        final_dict['filename'].append(str(os.path.basename(result_dict['filename'])))
        final_dict['size'].append(str(result_dict['size']))
        final_dict['query'].append(str(result_dict['query']))
        final_dict['answer'].append(str(result_dict['answer']))
        final_dict['pred'].append(str(result_dict['pred']))
        final_dict['answer_bbox'].append(str(answer_bbox))
        final_dict['pred_bbox'].append(str(pred_bbox))
        final_dict['pred_bbox_ori'].append(str(pred_bbox_ori))
        final_dict['iou'].append(iou)
        
    precision = np.sum(level_hit_count) / sum(level_count)
    perf_dict = {
        'precision': precision
    }
    ret.update({'precision': precision})
    level_precision = level_hit_count / level_count
    for i, name in enumerate(LEVEL_NAME):
        ret.update({f'precision_{name}_{level_count[i]}': level_precision[i]})
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    dump(ret, save_excel)

    print(f'precision: {precision}')
    for i, name in enumerate(LEVEL_NAME):
        print(f'precision_{name}: {level_precision[i]}')
    return perf_dict


def eval_results_ml(args, result_json_file, save_excel):


    with open(result_json_file, 'r') as f:
        result_lines = f.readlines()
    
    def extract_cate(text):
        start_index = text.find(':')
        cate_names = [name.strip() for name in text[start_index+1:-1].split(',')]
        return cate_names
    
    def match_cate(text, cate_names):
        match_index = []
        for cate_name in cate_names:
            if cate_name in text:
                match_index.append(1)
            else:
                match_index.append(0)
        return match_index

    dataset_size = len(result_lines)
    ret = {}
    final_dict = defaultdict(list)

    first_dict = json.loads(result_lines[0])
    cate_names = extract_cate(str(first_dict['query']))
    answer_list = []
    pred_list = []
    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        h, w = result_dict['size']
        assert cate_names == extract_cate(str(result_dict['query']))
        answer_index_array = np.array(match_cate(str(result_dict['answer']), cate_names))
        pred_index_array = np.array(match_cate(str(result_dict['pred']), cate_names))
        answer_list.append(answer_index_array)
        pred_list.append(pred_index_array)
        answer_index = np.nonzero(answer_index_array)[0]
        pred_index = np.nonzero(pred_index_array)[0]

        instance_op = len(np.nonzero((pred_index_array == answer_index_array) & (pred_index_array == 1))[0]) / np.sum(pred_index_array)
        
        final_dict['filename'].append(str(os.path.basename(result_dict['filename'])))
        final_dict['size'].append(str(result_dict['size']))
        final_dict['query'].append(str(result_dict['query']))
        final_dict['answer'].append(str(result_dict['answer']))
        final_dict['pred'].append(str(result_dict['pred']))
        final_dict['answer_cate'].append(str([cate_names[i] for i in answer_index]))
        final_dict['pred_cate'].append(str([cate_names[i] for i in pred_index]))
        final_dict['instance_op'].append(str(instance_op))

    answer_label = np.vstack(answer_list)
    pred_label = np.vstack(pred_list)

    cp = np.sum(((answer_label == pred_label) & (pred_label == 1)).astype(int), axis=0) / np.sum(pred_label, axis=0)
    average_CP = np.nan_to_num(cp).mean()

    perf_dict = {'average CP': average_CP}
    ret.update(perf_dict)
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    dump(ret, save_excel)

    print(f'average CP: {average_CP}')
    return perf_dict


def eval_results_mae(args, result_json_file, save_excel):


    with open(result_json_file, 'r') as f:
        result_lines = f.readlines()

    dataset_size = len(result_lines)
    ret = {}
    final_dict = defaultdict(list)

    global_MAE_list = []
    global_MAPE_list = []
    parsing_failure = 0
    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        h, w = result_dict['size']
        
        prediction = str(result_dict['pred'])
        answer = str(result_dict['answer'])
        
        answer_numbers = EXTRACT_NUMBER_PATTERN.findall(answer)
        prediction_numbers = EXTRACT_NUMBER_PATTERN.findall(prediction)
        # ====== calculate MAE and MAPE ======
        if len(answer_numbers) == len(prediction_numbers):
            for j in range(len(answer_numbers)):
                global_MAE_list.append(abs(float(prediction_numbers[j])-float(answer_numbers[j])))
                global_MAPE_list.append(abs((float(prediction_numbers[j])-float(answer_numbers[j])) / float(answer_numbers[j])))
        else:
            parsing_failure += 1
        
        final_dict['filename'].append(str(os.path.basename(result_dict['filename'])))
        final_dict['size'].append(str(result_dict['size']))
        final_dict['query'].append(str(result_dict['query']))
        final_dict['answer'].append(str(result_dict['answer']))
        final_dict['pred'].append(str(result_dict['pred']))
        final_dict['answer_numbers'].append(str(answer_numbers))
        final_dict['pred_numbers'].append(str(prediction_numbers))
        if len(answer_numbers) == len(prediction_numbers):
            instance_MAE = sum(global_MAE_list[-len(answer_numbers):]) / len(answer_numbers)
            instance_MAPE = sum(global_MAPE_list[-len(answer_numbers):]) / len(answer_numbers)
        else:
            instance_MAE = None
            instance_MAPE = None
        final_dict['instance_MAE'].append(str(instance_MAE))
        final_dict['instance_MAPE'].append(str(instance_MAPE))

    MAE = sum(global_MAE_list) / len(global_MAE_list)
    MAPE = sum(global_MAPE_list) / len(global_MAPE_list)
    Parsing_Failure = parsing_failure
    Parsing_Failure_ratio = parsing_failure / dataset_size

    perf_dict = {
        'MAE': MAE,
        'MAPE': MAPE,
    }
    ret.update({
        'MAE': MAE,
        'MAPE': MAPE,
        'Parsing_Failure': Parsing_Failure,
        'Parsing_Failure_ratio': Parsing_Failure_ratio,
    })
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    dump(ret, save_excel)

    print(f'MAE: {MAE}')
    return perf_dict


def eval_results_ciou(args, result_json_file, save_excel):
    from pycocotools import mask as cocomask

    with open(result_json_file, 'r') as f:
        result_lines = f.readlines()

    def extract_points(text):
        start_index = text.find('[')
        end_index = text.rfind(']')
        if start_index != -1 and end_index != -1:
            answer_numbers = EXTRACT_NUMBER_PATTERN.findall(text[start_index:end_index+1])
            return [float(number) for number in answer_numbers]
        else:
            return None

    def calc_IoU(a, b):
        i = np.logical_and(a, b)
        u = np.logical_or(a, b)
        I = np.sum(i)
        U = np.sum(u)

        iou = I/(U + 1e-9)

        is_void = U == 0
        if is_void:
            return 1.0
        else:
            return iou

    dataset_size = len(result_lines)
    ret = {}
    final_dict = defaultdict(list)

    list_iou = []
    list_ciou = []
    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        h, w = result_dict['size']
        
        pred_points_ori = extract_points(str(result_dict['pred']))
        answer_points_ori = extract_points(str(result_dict['answer']))
        
        answer_points_x = [float(point * w / 1000.0) for point in answer_points_ori[::2]]
        answer_points_y = [float(point * w / 1000.0) for point in answer_points_ori[1::2]]
        answer_points = [answer_points_x[i//2] if i % 2 == 0 else answer_points_y[i//2] for i in range(len(answer_points_ori))]

        rle = cocomask.frPyObjects([answer_points], h, w)
        m = cocomask.decode(rle)
        mask_gti = m.reshape((h, w))
        N_GT = len(answer_points) // 2
        mask_gti = mask_gti != 0

        if pred_points_ori:
            pred_points_x = [float(point * w / 1000.0) for point in pred_points_ori[::2]]
            pred_points_y = [float(point * w / 1000.0) for point in pred_points_ori[1::2]]
            pred_points = [pred_points_x[i//2] if i % 2 == 0 else pred_points_y[i//2] for i in range(len(pred_points_ori))]
            
            rle = cocomask.frPyObjects([pred_points], h, w)
            m = cocomask.decode(rle)
            mask = m.reshape((h, w))
            N = len(pred_points) // 2
            mask = mask != 0

            ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)
            iou = calc_IoU(mask, mask_gti)
            list_iou.append(iou)
            list_ciou.append(iou * ps)
        else:
            list_iou.append(None)
            list_ciou.append(None)

        final_dict['filename'].append(str(os.path.basename(result_dict['filename'])))
        final_dict['size'].append(str(result_dict['size']))
        final_dict['query'].append(str(result_dict['query']))
        final_dict['answer'].append(str(result_dict['answer']))
        final_dict['pred'].append(str(result_dict['pred']))
        final_dict['answer_points_ori'].append(str(answer_points_ori))
        final_dict['answer_points'].append(str(answer_points))
        final_dict['pred_points_ori'].append(str(pred_points_ori))
        final_dict['pred_points'].append(str(pred_points))
        final_dict['instance_iou'].append(str(list_iou[-1]))
        final_dict['instance_ciou'].append(str(list_ciou[-1]))

    mean_iou = np.mean(list_iou)
    mean_ciou = np.mean(list_ciou)

    perf_dict = {
        'mean_iou': mean_iou,
        'mean_ciou': mean_ciou,
    }
    ret.update(perf_dict)
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    dump(ret, save_excel)

    print(f'mean_iou: {mean_iou}')
    print(f'mean_ciou: {mean_ciou}')
    return perf_dict


def infer_single(model, anns_json_path, anns, task_type):
    fn = anns['image']
    if os.path.isabs(anns['image_path']):
        fn_full = os.path.join(anns['image_path'],fn)
    else:
        dataset_base = os.path.dirname(anns_json_path)
        fn_full = os.path.join(dataset_base, anns['image_path'], fn)

    q,a = convt_qa(anns['conversations'], task_type)
    q = q.replace('<image>\n','')
    if 'size' not in anns.keys():
        image = Image.open(fn_full).convert('RGB')
        anns['size'] = image.size
    result_dict = {'filename': fn_full, 'size': anns['size'], 'query': q, 'answer': a}
    
    try:
        with torch.inference_mode():
            outputs = model.generate(fn_full, q)
        # print('pred:', outputs)
        result_dict['pred'] = outputs
    except Exception as e:
        print(f'An error occurred: {e}')
        result_dict['pred'] = str(e)
    return result_dict


def infer_model(args, model, anns_json_path, task_type, save_json, local_rank, world_size):
    with open(anns_json_path, 'r') as f:
        anns_dict = json.load(f)
    
    chunk_size = len(anns_dict) // world_size
    sub_lists = [anns_dict[i:i + chunk_size] for i in range(0, len(anns_dict), chunk_size)]
    if len(anns_dict) % world_size != 0:
        sub_lists[-2] = sub_lists[-2] + sub_lists[-1]
    sub_anns_dict = sub_lists[local_rank]

    # count = 0
    final_results = []
    for idx, anns in tqdm(enumerate(sub_anns_dict), total=len(sub_anns_dict)):
        # print(idx,'/',len(anns_dict))
        result_dict = infer_single(model, anns_json_path, anns, task_type)

        final_results.append(json.dumps(result_dict))
        # count += 1
        # if count > 8:
        #     break

    if world_size > 1:
        dist.barrier()
        if local_rank == 0:
            # Rank 0 will collect objects from all processes, including itself
            gathered_objects = []
            for _ in range(dist.get_world_size()):
                gathered_objects.append(None)  # Placeholder for the incoming objects
            dist.gather_object(final_results, object_gather_list=gathered_objects, dst=0)

            final_results = []
            for sublist in gathered_objects:
                final_results.extend(sublist)
        else:
            # Other ranks send their objects to rank 0
            dist.gather_object(final_results, dst=0)

    if local_rank == 0:
        with open(save_json, 'w') as f:
            f.write('\n'.join(final_results))



class VHMDataset(torch.utils.data.Dataset):

    def __init__(self, root, prompt, image_root, input_size=448, dynamic_image_size=False,
                 use_CoT=False,use_thumbnail=False, max_num=6):
        data=[]
        with jsonlines.open(root, "r") as reader:
            for obj in reader:
                data.append(obj)
            self.ann_data = data
        # with open(root, 'r') as f:
        #     self.ann_data = json.load(f)
        self.prompt = prompt
        self.image_root = image_root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.use_CoT=use_CoT
    def __len__(self):
        return len(self.ann_data)

    def __getitem__(self, idx):
        data_item = self.ann_data[idx]
        # index = data_item['id']
        image = data_item['image']
        # question = data_item['question'] + '\n' + self.prompt
        # answer = data_item['gt_answer']
        question = data_item['conversations'][0]['value']
        if self.use_CoT:
            question=question.split(': ')[-1].split('. ')[0]
            question='<image>\n<cls>Please reason and then identify the category or categories for this image by selecting one or more from: '+question
        answer = data_item['conversations'][1]['value']
        # question_type = data_item['type']
        question_type = data_item['id'].split('_')[0]
        image = Image.open(os.path.join(self.image_root, image)).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'question': question,
            'pixel_values': pixel_values,
            'answer': answer,
            # 'index': index,
            'filename': data_item['image'],
            'question_type': 'cls'
        }
def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    filename= [_['filename'] for _ in batches]
    question_type = [_['question_type'] for _ in batches]
    # image_sizes = [_['image_size'] for _ in batches]

    return pixel_values, questions, answers,filename,question_type#, image_sizes
class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
def evaluate_chat_model():
    if args.seed!=-1:
        random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = VHMDataset(
            root=ds_collections[ds_name]['root'],
            prompt='',#prompt,
            image_root=ds_collections[ds_name]['image_root'],
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
            use_CoT=args.cot
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        for _, (pixel_values, questions, answers, images, question_types) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            preds = []
            for _ in range(args.repeat_times):
                pred = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=questions[0],
                    generation_config=generation_config
                )
                preds.append(pred.lower())
                counter = Counter(preds)
                pred, most_common_count = counter.most_common(1)[0]
            for question, preds_, pred, answer, image, question_type in zip(questions, [preds], [pred], answers, images, question_types):
                outputs.append({
                    'question': question,
                    'preds': preds_,
                    'pred': pred,
                    'answer': answer,
                    'image': image,
                    # 'index': int(index),
                    'question_type': question_type
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            if args.cot:
                results_file = f'{ds_name}_{time_prefix}_COT.json'
            else:
                results_file = f'{ds_name}_{time_prefix}.json'
                
            output_path = os.path.join(args.out_dir, f_dir , results_file)
            with open(output_path, 'w') as f:
                json.dump(merged_outputs, f, indent=4)
            ds_=ds_name.split('_CoT')[0]
            cmd = f'python eval/vhm/score.py --dataset {ds_} --output_file {output_path}'
            print(cmd)
            os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='cls_AID')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--repeat-times', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    f_dir=args.checkpoint.split('/')[-1]
    if not os.path.exists(os.path.join(args.out_dir,f_dir)):
        os.makedirs(os.path.join(args.out_dir,f_dir), exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)

    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-name", type=str, default="vhm")
    # parser.add_argument("--conv-mode", type=str, default="v1")
    # parser.add_argument("--task", type=str)
    # parser.add_argument("--dataset-base", type=str, default=r'/mnt/petrelfs/pangchao/ljy_home_files/experiments/model_eval_new/datasets')
    # parser.add_argument("--save-path", type=str, default=r'/mnt/petrelfs/pangchao/ljy_home_files/experiments/model_eval_new/temp')
    # parser.add_argument("--force-inference", action='store_true')
    # parser.add_argument("--batch-per-gpu", type=int, default=3)
    # args = parser.parse_args()

    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # world_size = int(os.environ.get("WORLD_SIZE", 1))

    # if world_size > 1:
    #     assert world_size % args.batch_per_gpu == 0
    #     torch.cuda.set_device(local_rank // args.batch_per_gpu)
    #     # dist.init_process_group(backend='nccl', world_size=world_size)
    #     dist.init_process_group(backend='gloo', world_size=world_size)

    # Model
    # disable_torch_init()
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    # model = VHM(name=args.model_name, model_path_map=model_path_map)

    # if local_rank == 0:
    #     if not os.path.exists(args.save_path):
    #         os.makedirs(args.save_path)
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')
    print(f'[test] max_num: {args.repeat_times}')

    # prompt = 'Answer the question using a single word or phrase.'
    evaluate_chat_model()

    # total_perf_dict = {}
    # if args.task == 'all':
    #     for task_index in BENCH_DATASETS.keys():
    #         if local_rank == 0:
    #             print(f'===================={task_index}======================')
    #         perf_dict = eval_task(args, model, task_index, local_rank, world_size)
    #         total_perf_dict.update({task_index: perf_dict})
    # else:
    #     perf_dict = eval_task(args, model, args.task, local_rank, world_size)
    #     total_perf_dict.update({args.task: perf_dict})
    
    # # Generate performance profile
    # if local_rank == 0:
    #     # breakpoint()
    #     total_perf_dict = {k: v if v is not None else {} for k, v in total_perf_dict.items()}
    #     df = pd.DataFrame.from_dict(total_perf_dict, orient='index')
    #     df.to_excel(os.path.join(args.save_path, 'profile.xlsx'))
