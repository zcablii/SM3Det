import os
import json
import math
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, Subset

from .models import Florence2Processor
from .models.florence2.modeling_florence2 import shift_tokens_right


class OrientedDetSFTDataset(Dataset):
    def __init__(self, processor, data_args, model=None):
        super(OrientedDetSFTDataset, self).__init__()
        self.data_args = data_args
        self.processor = processor
                
        self.prompt = Florence2Processor.task_prompts_without_inputs['<ROD>']
        
        if self.data_args.dataset_mode == "single":
            assert len(data_args.data_path) == 1, "data_path should have only one element in single mode."
            assert len(data_args.image_folder) == 1, "image_folder should have only one element in single mode."
            self.list_data_dict = json.load(open(data_args.data_path[0], "r"))
            for data_dict in self.list_data_dict:
                data_dict["image_folder"] = data_args.image_folder[0]
        else:
            self.deal_with_multidataset_mode()
        
        if data_args.language_model_max_length is not None:
            self.max_length = data_args.language_model_max_length
        else:
            self.max_length = 1024

        if model is not None:
            model_class = type(model).__name__
            if model_class == "Florence2ForConditionalGeneration":
                self.decoder_start_token_id = model.config.text_config.decoder_start_token_id
                self.num_beams = model.config.text_config.num_beams
                self.pad_token_id = model.config.text_config.pad_token_id
                self.decoder_start_token = self.processor.tokenizer.decode(self.decoder_start_token_id)
                self.decoder_second_token = self.processor.tokenizer.decode(0)

    def deal_with_multidataset_mode(self, epoch=0):
        # concat / balanced concat
        assert len(self.data_args.data_path) == len(self.data_args.image_folder), "data_path and image_folder should have the same length."
        if self.data_args.dataset_mode == "concat":
            if not hasattr(self, "list_data_dict"):
                self.list_data_dict = []
                for data_path, image_folder in zip(self.data_args.data_path, self.data_args.image_folder):
                    list_data_dict = json.load(open(data_path, "r"))
                    for data_dict in list_data_dict:
                        data_dict["image_folder"] = image_folder
                    self.list_data_dict.extend(list_data_dict)
        
        elif self.data_args.dataset_mode == "balanced concat":
            if not (
                hasattr(self, "balanced_subsets") and \
                hasattr(self, "dummy_dataset_len") and \
                hasattr(self, "dummy_subset_len")
            ):
                subsets = []
                for data_path, image_folder in zip(self.data_args.data_path, self.data_args.image_folder):
                    list_data_dict = json.load(open(data_path, "r"))
                    for data_dict in list_data_dict:
                        data_dict["image_folder"] = image_folder
                    subsets.append(list_data_dict)

                len_subset = [len(subset) for subset in subsets]
                self.dummy_dataset_len = sum(len_subset)
                self.dummy_subset_len = math.ceil(self.dummy_dataset_len / len(subsets))

                subsets = [subset * math.ceil(self.dummy_subset_len / len(subset)) for subset in subsets]
                self.balanced_subsets = subsets

            print(f"making list_data_dict with seed {epoch}")
            rnd = random.Random(epoch)
            self.list_data_dict = []
            for subset in self.balanced_subsets:
                self.list_data_dict.extend(rnd.sample(subset, self.dummy_subset_len))
            rnd.shuffle(self.list_data_dict)
            self.list_data_dict = self.list_data_dict[:self.dummy_dataset_len]
        elif self.data_args.dataset_mode == "single":
            pass
        else:
            raise ValueError(f"dataset_mode={self.data_args.dataset_mode} is not supported.")
        
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i):
        data = self.list_data_dict[i]
        pil_img = Image.open(os.path.join(data["image_folder"], data["file_name"])).convert('RGB')
        response = data["response"]
        if self.data_args.response_format == "allseeing":
            response = self.florence2fmt_to_allseeingfmt(response)
        return pil_img, response
    
    def collate_fn(self, batch_data):
        model_type = self.data_args.model_type

        collate_fn_name = f"_collate_fn_{model_type}"
        if hasattr(self, collate_fn_name):
            return getattr(self, collate_fn_name)(batch_data)
        else:
            raise ValueError(f"model_type={self.data_args.model_type} is not supported.")
        
    def _collate_fn_florence2(self, batch_data):
        pil_imgs, responses = zip(*batch_data)

        self.prompt = self.processor.task_prompts_without_inputs['<ROD>']
        encoder_inputs = self.processor(
            text=[self.prompt] * len(pil_imgs), 
            images=pil_imgs, 
            return_tensors="pt", 
            padding=True,
        )
        
        response_prefix = self.decoder_second_token * max(0, self.num_beams - 2)
        responses = [response_prefix + response for response in responses]
        decoder_inputs = self.processor.tokenizer(
            text=responses, 
            return_tensors="pt", 
            padding="longest", 
            padding_side="right",
            truncation=True,
            return_token_type_ids=False,
            max_length=self.max_length,
        )

        labels = decoder_inputs["input_ids"][:, 1:]  # exclude bos tokens
        decoder_attention_mask = decoder_inputs["attention_mask"]
        labels.masked_fill_(~decoder_attention_mask.bool()[:, 1:], -100)
        labels = labels.contiguous()
        decoder_attention_mask = decoder_attention_mask[:, :-1]  # treate the position of bos tokens as self.decoder_start_token_id
        decoder_input_ids = shift_tokens_right(
            labels, 
            pad_token_id=self.pad_token_id, 
            decoder_start_token_id=self.decoder_start_token_id
        )
        
        return {
            "input_ids": encoder_inputs["input_ids"], 
            "pixel_values": encoder_inputs["pixel_values"], 
            "attention_mask": encoder_inputs["attention_mask"], 
            "decoder_input_ids": decoder_input_ids, 
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
    
    def _collate_fn_qwen2_vl(self, batch_data):
        pil_imgs, responses = zip(*batch_data)

        message_base = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]},
        ]

        templated_instruction = self.processor.apply_chat_template(message_base, add_generation_prompt=True)

        all_templated_instruction_and_reponse = []
        all_templated_response = []
        for response in responses:
            msg = message_base + [{"role": "assistant", "content": response}]
            templated_instruction_and_reponse = self.processor.apply_chat_template(msg).rstrip("\n")
            templated_response = templated_instruction_and_reponse.replace(templated_instruction, "")
            assert "".join([templated_instruction, templated_response]) == templated_instruction_and_reponse
            all_templated_instruction_and_reponse.append(templated_instruction_and_reponse)
            all_templated_response.append(templated_response)

        inputs = self.processor(
            text=[templated_instruction] * len(pil_imgs), 
            images=pil_imgs, 
            return_tensors="pt", 
            padding=True,
        )
        assert torch.all(inputs.attention_mask)
        inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

        additional_inputs = self.processor.tokenizer(
            text=all_templated_response, 
            return_tensors="pt", 
            padding="longest", 
            padding_side="right",
            truncation=True,
            return_token_type_ids=False,
            max_length=self.max_length,
        )
        additional_input_ids = additional_inputs["input_ids"]
        additional_attention_mask = additional_inputs["attention_mask"]
        additional_labels = additional_input_ids.clone()
        additional_labels[~additional_attention_mask.bool()] = -100

        inputs["input_ids"] = torch.cat([inputs["input_ids"], additional_input_ids], dim=1)
        inputs["labels"] = torch.cat([inputs["labels"], additional_labels], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], additional_attention_mask], dim=1)
        return inputs

    def _collate_fn_internvl2(self, batch_data):
        pil_imgs, responses = zip(*batch_data)

        inputs = self.processor(
            questions=[self.prompt] * len(pil_imgs), 
            answers=responses,
            images=pil_imgs, 
            return_tensors="pt", 
            padding=True,
            max_length=self.max_length,
        )
        return {**inputs}

    def _collate_fn_llava_qwen(self, batch_data):
        pil_imgs, responses = zip(*batch_data)

        inputs = self.processor(
            questions=[self.prompt] * len(pil_imgs), 
            answers=responses,
            images=pil_imgs, 
            max_length=self.max_length,
        )
        return {**inputs}

    def florence2fmt_to_allseeingfmt(self, text):
        if not hasattr(self, "florence2_post_processor"):
            from .models import Florence2PostProcesser
            self.florence2_post_processor = Florence2PostProcesser(tokenizer=self.processor.tokenizer)
        task_answer_post_processing_type = "description_with_polygons"

        task_answer = self.florence2_post_processor(
            text=text,
            image_size=(1000, 1000),
            parse_tasks=task_answer_post_processing_type,
        )[task_answer_post_processing_type]

        new_text = ""
        for result in task_answer:
            label = result['cat_name']
            polygons = [[int(value - 0.5) for value in polygon] for polygon in result['polygons']]  # TODO: there may be mismatch between box decoding and encoding
            new_text += f"<ref>{label}</ref><box>{polygons}</box>"
        return new_text

    
class OrientedDetEvalDataset:

    default_data_root = {
        "dota": "playground/data/split_ss_dota", 
        "dior": "playground/data/DIOR",
        "fair1m": "playground/data/split_ss_fair1m_1_0", 
        "fair1m_2.0_train": "playground/data/split_ss_fair1m_2_0",
        "srsdd": "playground/data/SRSDD",
        "dota_train": "playground/data/split_ss_dota",
        "rsar": "playground/data/RSAR",
    }
    
    func_map = {  # dataset_type: (is_test_set, not is_test_set)
        "dota": ("initialize_dota_dataset", "initialize_coco_format_dota_dataset"),
        "dior": ("initialize_coco_format_dior_dataset", "initialize_coco_format_dior_dataset"),
        "fair1m": ("initialize_fair1m_dataset", "initialize_coco_format_fair1m_dataset"),
        "fair1m_2.0_train": ("initialize_coco_format_fair1m_dataset", "initialize_coco_format_fair1m_dataset"),
        "srsdd": ("initialize_coco_format_srsdd_dataset", "initialize_coco_format_srsdd_dataset"),
        "dota_train": ("initialize_dota_dataset", "initialize_dota_dataset"),
        "rsar": ("initialize_rsar_dataset", "initialize_coco_format_rsar_dataset"),
    }

    def __init__(self, dataset_type="dota", data_root=None, shuffle_seed=42, clip_num=None, is_test_set=False):
        self.dataset_type = dataset_type
        self._data_root = data_root
        self.is_test_set = is_test_set
        
        from mmdet.utils import register_all_modules as register_mmdet_modules
        from mmrotate.utils import register_all_modules as register_mmrotate_modules
        register_mmdet_modules(init_default_scope=False)
        register_mmrotate_modules(init_default_scope=True)

        self.initialize_dataset()
        
        if clip_num is not None or shuffle_seed is not None:
            indices = list(range(len(self.dataset)))
            if shuffle_seed is not None:
                random.Random(shuffle_seed).shuffle(indices)
            if clip_num is not None:
                indices = indices[:clip_num]
            self.dataset = Subset(self.dataset, indices)
        
        self.cls_map = {c.replace("-", " ").lower(): i
            for i, c in enumerate(self.metainfo['classes'])
        }  # in mmdet v2.0 label starts from 0

    @property
    def data_root(self):
        return self._data_root or self.default_data_root[self.dataset_type]

    def initialize_dataset(self):
        func_name = self.func_map[self.dataset_type][int(not self.is_test_set)]
        getattr(self, func_name)()

    def initialize_dota_dataset(self):
        from mmrotate.datasets import DOTADataset
        online_pipeline = [
            dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name'))
        ]

        offline_pipeline = [
            dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
            dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name'))
        ]

        if self.dataset_type == "dota_train":
            img_prefix = 'trainval/images/'
            ann_file = 'val/annfiles/' if self.is_test_set else 'train/annfiles/'
            pipeline = offline_pipeline
        elif self.dataset_type == "dota":
            img_prefix = 'test/images/' if self.is_test_set else 'trainval/images/'
            ann_file = '' if self.is_test_set else 'trainval/annfiles/'
            pipeline = online_pipeline if self.is_test_set else offline_pipeline
        else:
            raise ValueError(f"dataset_type={self.dataset_type} is not supported.")
        
        self.dataset = DOTADataset(
            data_root=self.data_root, 
            ann_file=ann_file,
            data_prefix=dict(img_path=img_prefix),
            test_mode=True,
            pipeline=pipeline,
        )
        
    def initialize_coco_format_dota_dataset(self):
        "Faster initialized"
        assert not self.is_test_set, "COCO format is not implemented for test set here."
        classes=('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                 'basketball-court', 'storage-tank', 'soccer-ball-field',
                 'roundabout', 'harbor', 'swimming-pool', 'helicopter')
        self.dataset = self.initialize_coco_format_dataset(self.data_root, classes, ann_file='trainval.json', img_prefix='trainval/images/')

    def initialize_dior_dataset(self):
        from mmrotate.datasets import DIORDataset
        self.dataset = DIORDataset(
            data_root=self.data_root, 
            ann_file='ImageSets/Main/test.txt' if self.is_test_set else 'ImageSets/Main/trainval.txt',  # you may require `cat train.txt val.txt > trainval.txt` to generate this file
            data_prefix=dict(img_path='JPEGImages-test') if self.is_test_set else dict(img_path='JPEGImages'),
            test_mode=True,
            pipeline=[
                dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
                dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name'))
            ]
        )

    def initialize_coco_format_dior_dataset(self):
        classes = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
                   'chimney', 'expressway-service-area', 'expressway-toll-station',
                   'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
                   'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')
        ann_file = 'Annotations/test.json' if self.is_test_set else 'Annotations/trainval.json'
        img_prefix = 'JPEGImages-test/' if self.is_test_set else 'JPEGImages-trainval/'
        self.dataset = self.initialize_coco_format_dataset(self.data_root, classes, ann_file, img_prefix)

    def initialize_fair1m_dataset(self):
        from lmmrotate.modules.fair_dataset import FAIRDataset
        self.dataset = FAIRDataset(
            data_root=self.data_root, 
            ann_file='' if self.is_test_set else 'train/annfiles/',
            data_prefix=dict(img_path='test/images/') if self.is_test_set else dict(img_path='train/images/'),
            test_mode=True,
            pipeline=[
                dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name'))
            ] if self.is_test_set else [
                dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
                dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name'))
            ]
        )

    def initialize_coco_format_fair1m_dataset(self):
        if self.is_test_set:
            if self.dataset_type == "fair1m_2.0_train":
                ann_file = 'validation/val.json'
                img_prefix = 'validation/images/'
            else:
                assert not self.is_test_set, "COCO format is not implemented for test set here."
        else:
            ann_file = 'train/train.json'
            img_prefix = 'train/images/'
        classes = ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
                   'A321', 'A330', 'A350', 'ARJ21', 'Passenger Ship', 'Motorboat',
                   'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship',
                   'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
                   'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator',
                   'Truck Tractor', 'Basketball Court', 'Tennis Court', 'Football Field',
                   'Baseball Field', 'Intersection', 'Roundabout', 'Bridge')
        self.dataset = self.initialize_coco_format_dataset(self.data_root, classes, ann_file=ann_file, img_prefix=img_prefix)
    
    def initialize_coco_format_srsdd_dataset(self):
        classes = ('Cell-Container', 'Container', 'Dredger', 'Fishing', 'LawEnforce', 'ore-oil')
        ann_file = 'test.json' if self.is_test_set else 'train.json'
        img_prefix = 'test/images/' if self.is_test_set else 'train/images/'
        self.dataset = self.initialize_coco_format_dataset(self.data_root, classes, ann_file, img_prefix)

    def initialize_rsar_dataset(self):
        from lmmrotate.modules.rsar_dataset import RSARDataset
        self.dataset = RSARDataset(
            data_root=self.data_root, 
            ann_file='test/annfiles/' if self.is_test_set else 'trainval/annfiles/',  # you may require `cat train.txt val.txt > trainval.txt` to generate this file
            data_prefix=dict(img_path='test/images/') if self.is_test_set else dict(img_path='trainval/images/'),
            test_mode=True,
            pipeline=[
                dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
                dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name'))
            ]
        )

    def initialize_coco_format_rsar_dataset(self):
        classes = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
        ann_file = 'test.json' if self.is_test_set else 'trainval.json'
        img_prefix = 'test/images/' if self.is_test_set else 'trainval/images/'
        self.dataset = self.initialize_coco_format_dataset(self.data_root, classes, ann_file, img_prefix)

    @staticmethod
    def initialize_coco_format_dataset(data_root, classes, ann_file, img_prefix):
        from mmdet.datasets import CocoDataset
        from mmdet.datasets.transforms import LoadAnnotations

        class CustomLoadAnnotations(LoadAnnotations):
            def transform(self, results):
                results["ori_shape"] = results['height'], results['width']
                results["file_name"] = os.path.basename(results["img_path"])
                results = super().transform(results)
                return results

        return CocoDataset(
            data_root=data_root, 
            metainfo=dict(classes=classes),
            ann_file=ann_file,
            data_prefix=dict(img=img_prefix),
            test_mode=True,
            pipeline=[
                CustomLoadAnnotations(with_bbox=True, with_mask=True, poly2mask=False),
                dict(type='ConvertMask2BoxType', box_type='rbox'),
                dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'file_name'))
            ]
        )
    
    @property
    def metainfo(self):
        return getattr(self.dataset, "metainfo", self.dataset.dataset.metainfo)

    def __getitem__(self, idx):
        data_sample = self.dataset[idx]["data_samples"]
        img_path = data_sample.img_path
        return img_path, data_sample
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
