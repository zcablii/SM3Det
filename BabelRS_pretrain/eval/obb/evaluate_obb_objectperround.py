import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import jsonlines
import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
import datetime
os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout



ds_collections = {
    'DOTA': {
        # 'root' : '../../InternRS_data/val_annos/obb_rgb_dota_multiround_test.jsonl',
        'root' : '../../InternRS_data/val_annos/obb_rgb_dota_multiround_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root' : '../../InternRS_data/',
        'batch_size' :1,
        # 'classes':('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        #          'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        #          'basketball-court', 'storage-tank', 'soccer-ball-field',
        #          'roundabout', 'harbor', 'swimming-pool', 'helicopter')
    },
    'DOTA_100': {
        'root' : '../../InternRS_data/val_annos/obb_rgb_dota_test_100.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root' : '../../InternRS_data/',
        'batch_size' :1,
        # 'classes':('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        #          'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        #          'basketball-court', 'storage-tank', 'soccer-ball-field',
        #          'roundabout', 'harbor', 'swimming-pool', 'helicopter')
    },
    'FAIR1M2': {
        'root' : '../../InternRS_data/val_annos/obb_rgb_fair1m2_multiround_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root' : '../../InternRS_data/',
        'batch_size' :1,
        # 'classes' : ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
        #            'A321', 'A330', 'A350', 'ARJ21', 'Passenger Ship', 'Motorboat',
        #            'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship',
        #            'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
        #            'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator',
        #            'Truck Tractor', 'Basketball Court', 'Tennis Court', 'Football Field',
        #            'Baseball Field', 'Intersection', 'Roundabout', 'Bridge')
    },
    'RSAR': {
        'root' : '../../InternRS_data/val_annos/obb_sar_rsar_multiround_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root' : '../../InternRS_data/',
        'batch_size' :1,
        # 'classes' : ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
    },
    'SRSDD': {
        'root' : '../../InternRS_data/val_annos/obb_sar_srsdd_multiround_test.jsonl',
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root' : '../../InternRS_data/',
        'batch_size' :1,
        # 'classes' : ('Cell-Container', 'Container', 'Dredger', 'Fishing', 'LawEnforce', 'ore-oil')
    },
    'MillionAID': {
        'root' : '../../InternRS_data/val_annos/MillionAID_test.jsonl',
        'max_new_tokens': 6000,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root' : '../../InternRS_data/val_dataset/',
        # 'classes' : ('Cell-Container', 'Container', 'Dredger', 'Fishing', 'LawEnforce', 'ore-oil')
    },
}


def collate_fn(batches, tokenizer):
    
    pixel_values = torch.cat([_['pixel_values'] for _ in batches])
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    image_names=[_['image_name'] for _ in batches]
    # image_sizes = [_['image_size'] for _ in batches]
    num_patches_list=[_['pixel_values'].shape[0] for _ in batches]
    return pixel_values, questions, answers, image_names,num_patches_list


class GroundingDataset(torch.utils.data.Dataset):

    def __init__(self, root, image_root, prompt='', input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        data=[]
        with jsonlines.open(root, "r") as reader:
            for obj in reader:
                data.append(obj)
            self.ann_data = data
        # with open(result_json_file, 'r') as f:
        #     result_lines = f.readlines()
        # with open(root, 'r') as f:
        #     self.ann_data = json.load(f)
        self.image_root = image_root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.prompt = prompt

    def __len__(self):
        return len(self.ann_data)

    def __getitem__(self, idx):
        data_item = self.ann_data[idx]
        # index = data_item["id"]
        image = data_item['image']
        question=[]
        answer=[]
        for i in range(0, len(data_item['conversations']), 2):

            question.append(data_item['conversations'][i]['value'])#self.prompt
            answer.append(data_item['conversations'][i+1]['value'])
        image_name = data_item['image']#.split('/')[-1]
        # image_size_=0
        # catetory = self.df.iloc[idx]['category']
        # l2_catetory = self.df.iloc[idx]['l2-category']
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
            'image_name': image_name
        }

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
def merge_and_save(outputs,ds_name,prefix,part_id,time_prefix):
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        # part_id=part_id+1
        f_dir=args.checkpoint.split('/')[-1]
        results_file = f'{prefix}_{ds_name}_{time_prefix}_{part_id}.json'
        output_path = os.path.join(args.out_dir,f_dir, results_file)
        # print('merged_outputs',merged_outputs)
        with open(output_path, 'w') as f:
            json.dump({'outputs': merged_outputs}, f, indent=4)
        print('Results saved to {}'.format(output_path))
        return output_path
    return None

def merge_partial_files(ds_name,all_files,prefix,time_str):
    """合并部分文件"""
    output_dir = os.path.join(args.out_dir, args.checkpoint.split('/')[-1])
    # all_files = sorted([f for f in os.listdir(output_dir) if f.startswith(f'{prefix}_{ds_name}')])
    
    merged_data = []
    for filename in all_files:
        with open(filename, 'r') as f:
            data = json.load(f)
            merged_data.extend(data['outputs'])
        # os.remove(filename)
    
    # time_str = time.strftime('%Y%m%d%H%M%S')
    final_path = os.path.join(output_dir, f'{prefix}_{ds_name}_{time_str}.json')
    with open(final_path, 'w') as f:
        json.dump({'outputs': merged_data}, f, indent=4)
    return final_path

def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = GroundingDataset(
            root=ds_collections[ds_name]['root'],
            image_root=ds_collections[ds_name]['image_root'],
            prompt='',#prompt_prefix+' '+('|').join(ds_collections[ds_name]['classes'])+'.',
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=ds_collections[ds_name]['batch_size'],
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        time_prefix = str(time.strftime('%y%m%d%H%M%S', time.localtime()))
        os.makedirs(os.path.join(args.out_dir,f_dir,'temp'), exist_ok=True)
        outputs = []
        # part_id=0
        cal_score_outputs= []
        readable_output_paths=[]
        cal_output_paths=[]
        for batch_id, (pixel_values, questions, answers, image_names,num_patches_list) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            preds=[]
                # print(pixel_values.shape[0])
            history=None
            # print(len(questions))
            # print(len(answers))
            questions=questions[0]
            answers=answers[0]
            for i, (question,answer) in enumerate(zip(list(questions),list(answers))):
                # print(question)
                pred = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_config,
                    # history=history,
                    # return_history=True
                )
                preds.append(pred)

            # for question, pred, answer, image_name_ in zip(questions, preds, answers, image_names):
            #     # print(answers)
            #     outputs.append({
            #         'question': question,
            #         'answer': pred,
            #         'gt_answers': answer,
            #         'image_name': image_name_
            #     })
            cal_score_outputs.append(
            {
                'answer': '|||'.join(preds),
                'gt_answers': answers,
                'image_name': image_names,
            })
            # print(batch_id)
            if (batch_id+1)%20==0:
                print('saving_ckpt')
                # part_id=part_id+1
                torch.distributed.barrier()
                # print(cal_score_outputs)
                # readable_output_paths.append(merge_and_save(outputs,ds_name,'temp/readable',batch_id,time_prefix))
                cal_output_paths.append(merge_and_save(cal_score_outputs,ds_name,'temp/cal',batch_id,time_prefix))
                torch.distributed.barrier()
                outputs=[]
                cal_score_outputs=[]

        # part_id=part_id+1
        torch.distributed.barrier()
        # readable_output_paths.append(merge_and_save(outputs,ds_name,'temp/readable','last',time_prefix))
        cal_output_paths.append(merge_and_save(cal_score_outputs,ds_name,'temp/cal','last',time_prefix))   
        
        
        if torch.distributed.get_rank() == 0:
            # merge_partial_files(ds_name,readable_output_paths,'readable',time_prefix)
            output_path=merge_partial_files(ds_name,cal_output_paths,'cal',time_prefix)
            print(f'Evaluating {ds_name} ...')
            cmd = f'python eval/obb/score.py --dataset {ds_name} --output_file {output_path}'
            #cmd = f'python eval/rs_det/caculate.py --output_file {output_path}'
            print(cmd)
            os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='DIOR_RSVG')
    # parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    f_dir=args.checkpoint.split('/')[-1]
    if not os.path.exists(os.path.join(args.out_dir,f_dir)):
        os.makedirs(os.path.join(args.out_dir,f_dir), exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    # assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        timeout=datetime.timedelta(seconds=72000000),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

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

    # prompt_prefix = '<image>\n<det><cls>Detect all objects with an orientated bounding box from the following categories:'
    # prompt_prefix =  "Please provide the bounding box coordinate of the region this sentence describes: "
    evaluate_chat_model()
