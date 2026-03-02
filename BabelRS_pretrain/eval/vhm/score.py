import argparse
import json
from rapidfuzz import process
import re
ds_collections = {
    'cls_AID': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_AID.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'cls',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes':["railway station", "beach", "industrial", "center", "storage tanks", 
            "meadow", "mountain", "airport", "resort", "school", "port", "farmland", "medium residential", 
            "pond", "river", "sparse residential", "forest", "playground", "church", "parking", "viaduct", 
            "stadium", "square", "bridge", "desert", "dense residential", "commercial", "park", "bare land", "baseball field","none"]
    },
    'cls_METER_ML': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_METER_ML.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes': [  "landfills", "coal mines", "oil refineries and petroleum terminals", 
            "natural gas processing plants", "wastewater treatment plants","concentrated animal feeding operations"]
    },
    'cls_NWPU_RESISC45': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_NWPU_RESISC45.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes':["airplane", "desert", "mobile home park", "ship", "airport", "forest", "mountain", "snowberg", "baseball diamond",
            "freeway", "overpass", "sparse residential", "basketball court", "golf course", "palace", "stadium", "beach", "ground track field",
            "parking lot", "storage tank", "bridge", "harbor", "railway", "tennis court", "chaparral", "industrial area", "railway station",
            "terrace", "church", "intersection", "rectangular farmland", "thermal power station", "circular farmland", "island", "river",
            "wetland", "cloud", "lake", "roundabout", "commercial area", "meadow", "runway", "dense residential", "medium residential", "sea ice"]
    },
    'cls_SIRI_WHU': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_SIRI_WHU.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes': ["river", "water", "agriculture", "harbor", "commercial", "overpass", "meadow", "industrial",
         "residential", "idle land", "pond", "park"]
    },
    'cls_WHU_RS19': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_WHU_RS19.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes': ["industrial", "park", "farmland", "river", "residential", "forest", "airport", "football field", 
        "viaduct", "railway station", "parking", "desert", "pond", "meadow", "beach", "bridge", "mountain", "port", "commercial"]
    },
    
    'cls_WHU_RS19_hard': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_WHU_RS19_hard.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes': ["industrial", "park", "farmland", "river", "residential", "forest", "airport", "football field", 
        "viaduct", "railway station", "parking", "desert", "pond", "meadow", "beach", "bridge", "mountain", "port", "commercial"]
    },

    'cls_millionAID': {
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_rgb_Million-AID_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes': ['woodland', 'church', 'roundabout', 'lake', 'commercial area', 'arable land', 'oil field', 'pier', 
            'leisure land', 'factory area', 'mine', 'grassland', 'mining area', 'tennis court', 'special land', 'meadow',
            'baseball field', 'river', 'swimming pool', 'helipad', 'basketball court', 'greenhouse', 'rock land', 
            'solar power plant', 'parking lot', 'sparse shrub land', 'substation', 'mobile home park', 'commercial land',
            'unutilized land', 'island', 'viaduct', 'industrial land', 'agriculture land', 'intersection', 
            'ground track field', 'port area', 'water area', 'religious land', 'apartment', 'road', 'ice land', 
            'highway area', 'public service land', 'transportation land', 'airport area', 'paddy field', 'quarry',
            'golf course', 'detached house', 'railway area', 'dam', 'bare land', 'forest', 'wastewater plant', 
            'sports land', 'beach', 'bridge', 'terraced field', 'orchard', 'apron', 'railway', 'desert', 'stadium', 
            'storage tank', 'power station', 'works', 'train station', 'residential land', 'wind turbine', 
            'dry field', 'cemetery', 'runway', 'commercial','terrace','none','electric substation','parking','dry land','bareland']
    },
    'cls_sar_TenGeoP':{
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_sar_TenGeoP-SARwv_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes': ['internal wave','ocean eddy','offshore wind power','oil spill alike','wind streaks','micro convective cells',
                    'rain cells','rainfall','biological slicks','sea ice','iceberg','low wind area','atmospheric front','oceanic front', 'none']
    },
    'cls_sar_ISPRS':{
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_sar_ISPRS_SAR_classification_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes': ['internal wave','ocean eddy','offshore wind power','oil spill alike','wind streaks','micro convective cells',
                    'rain cells','rainfall','biological slicks','sea ice','iceberg','low wind area','oceanic or atmospheric front', 'none']
    },

    'cls_fmow':{
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_rgb_fmow_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes': ['military facility','border checkpoint','port','railway bridge','airport hangar','helipad','road bridge',
                    'flooded road','tower','zoo','factory or powerplant','single-unit residential',
                    'place of worship','crop field','lake or pond','nuclear powerplant','aquaculture',
                    'ground transportation station','construction site','hospital','storage tank',
                    'recreational facility','electric substation','car dealership','fountain','gas station','dam',
                    'space facility','airport','smokestack','water treatment facility','oil or gas facility',
                    'office building','park','lighthouse','shipyard','parking lot or garage','tunnel opening',
                    'toll booth','debris or rubble','wind farm','swimming pool','multi-unit residential','runway',
                    'stadium','amusement park','prison','surface mine','race track','golf course','educational institution',
                    'waste disposal','shopping mall','impoverished settlement','airport terminal','fire station',
                    'archaeological site','barn','solar farm','police station','interchange','burial site']
    },
    'cls_UCM':{
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_rgb_UCM.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes': ['dense residential','freeway','chaparral','harbor','airplane','sparse residential','storage tanks','beach','forest','river','buildings','baseball diamond','tennis court','medium residential','runway','mobile home park','intersection','agricultural','parking lot','golf course','overpass']
    },
    'cls_sar_Sentinel':{
        'root': '../../InternRS_data/val_annos/VHM_eval/cls_sar_Sentinel_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes': ['urban','barren land','grass land','agriculture land']
    },
    'mcq_test':{
        'root': '../../InternRS_data/val_annos/VHM_eval/mcq_merged_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': '../../InternRS_data/val_dataset/VHM_eval',
        'classes': [ 'A Color image','B False color image','C Panchromatic image','D SAR image' ]
    },
}

def eval_results_cls_vqa(data,classes):
    print(classes)
    # result_lines=data
    # with open(result_json_file, 'r') as f:
    #     result_lines = f.readlines()

    # dataset_size = len(result_lines)
    dataset_size=len(data)
    ret = {}
    # final_dict = defaultdict(list)
    score = 0
    class_score={}
    
    class_size={}
    for class_ in classes:
        class_ = class_.lower()
        class_score[class_]=0
        class_size[class_]=0
    
    for img_i, result_dict in enumerate(data):
        raw_prediction = str(result_dict['pred'])
        answers = str(result_dict['answer'])
        predictions = re.sub(r'[^\w:.]', ' ', raw_prediction)
        predictions = predictions.lower().strip().strip('.').split('.')
        # predictions = predictions.strip().split(':')[-1]
        # predictions = predictions.strip().split(' a ')[-1]
        # predictions = predictions.strip().split(' an ')[-1]
        # predictions = predictions.strip().split(' of ')[-1]
        # predictions = predictions.strip().split(' or ')
        result_dict['pred_cls']=predictions
        answers = answers.lower()
        for prediction in predictions:
            prediction=prediction.strip()
            if prediction not in classes:
                # print(process.extractOne(prediction.lower(), classes))
                fuzzy_matched_cat,fuzzy_match_score,_ = process.extractOne(prediction.lower(), classes)
                
                # if logger is None:
                if fuzzy_match_score>=70.0:
                    print(f"Fuzzy matched {prediction.lower()} to {fuzzy_matched_cat}")
                    # print(fuzzy_match_score)
                    prediction=fuzzy_matched_cat
        # answers=answers.split(' or ')
        sco=0
        for answer in [answers]:
            answer=answer.strip()
            class_size[answer]=class_size[answer]+1
            class_score[answer]=class_score[answer]+(answer in predictions)
            sco = sco+(answer in predictions)
        score=score+(sco>0)
        result_dict['result']=sco>0
        # score = score+(answer == prediction)
        # final_dict['score'].append(prediction in answer)
        
        # final_dict['filename'].append(str(result_dict['image']))#os.path.basename(result_dict['filename'])))
        # # final_dict['size'].append(str(result_dict['size']))
        # final_dict['question'].append(str(result_dict['question']))
        # final_dict['answer'].append(str(result_dict['answer']))
        # final_dict['pred'].append(str(prediction))
        # final_dict['raw_pred'].append(str(result_dict['pred']))

    avg_score =  score / dataset_size # sum(final_dict['score']) /len(final_dict['score'])
    print(f'perception: {avg_score}')
    class_avg_score={}
    for class_ in classes:
        class_ = class_.lower()
        class_avg_score[class_]=0. if class_size[class_]==0 else class_score[class_]/class_size[class_]
    perf_dict = {
        'perception': avg_score,
        'class_perception':class_avg_score
        # 'outputs': final_dict
    }
    # ret.update(perf_dict)
    # ret.update(final_dict)

    # ret = pd.DataFrame({x: ret[x] for x in ret})
    # dump(ret, save_excel)

    
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
def calculate_scores(data,dataset,task_type='cls'):
    if task_type == 'bbox':
        return eval_results_bbox(data)
    elif task_type == 'ml':
        return eval_results_ml(args, save_json, save_excel)
    elif task_type == 'mae':
        return eval_results_mae(args, save_json, save_excel)
    elif task_type == 'ciou':
        return eval_results_ciou(args, save_json, save_excel)
    elif task_type == 'presence':
        return eval_results_presence(args, save_json, save_excel)
    elif task_type in ('position', 'imgType', 'mc'):
        return eval_results_multichoice(args, save_json, save_excel)
    elif task_type == 'counting':
        return eval_results_couting(args, save_json, save_excel)
    elif task_type == 'color':
        return eval_results_color(args, save_json, save_excel)
    elif task_type in ('cls', 'vqa'):
        # print(ds_collections[args.dataset]['classes'])
        return eval_results_cls_vqa(data,ds_collections[args.dataset]['classes'])
    return None

# def calculate_scores(data):
#     type_counts = {}
#     type_correct = {}
#     for entry in data:
#         question_type = entry['question_type']
#         response = entry['response']
#         answer = entry['gt_answer']

#         if question_type not in type_counts:
#             type_counts[question_type] = 0
#             type_correct[question_type] = 0
#         type_counts[question_type] += 1

#         if question_type == 'count':
#             if is_correct_count(response, answer):
#                 type_correct[question_type] += 1
#         elif question_type == 'area':
#             if is_correct_area(response, answer):
#                 type_correct[question_type] += 1
#         else:
#             if response and response.lower() == answer.lower():
#                 type_correct[question_type] += 1

#     type_scores = {}
#     for question_type in type_counts:
#         score = type_correct[question_type] / type_counts[question_type]
#         type_scores[question_type] = round(score, 4)

#     total_correct = sum(type_correct.values())
#     total_count = sum(type_counts.values())
#     total_score = round(total_correct / total_count, 4) if total_count > 0 else 0.0

#     total_correct_useful = sum([v for k, v in type_correct.items() if k not in ['count', 'area']])
#     total_count_useful = sum([v for k, v in type_counts.items() if k not in ['count', 'area']])
#     total_score_useful = round(total_correct_useful / total_count_useful, 4) if total_count_useful > 0 else 0.0
#     print(f'{type_scores=}')
#     print(f'{total_score_useful=}')
#     return type_scores, total_score, total_score_useful, type_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--output_file', type=str, default='')
    args = parser.parse_args()

    with open(args.output_file, 'r') as f:
        data = json.load(f)
    if 'outputs' in data:
        data = data['outputs']

    results = calculate_scores(data,args.dataset)
    results['outputs'] = data
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)