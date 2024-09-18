from .dota import *
from .builder import ROTATED_DATASETS
from mmengine.dataset import BaseDataset
from typing import List, Optional

 
@ROTATED_DATASETS.register_module()
class Dota_Dataset(DOTADataset):
 
    CLASSES = ('small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100),  (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (0, 226, 252)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='le90',
                 difficulty=100,
                 cache_annotations=False,
                 cache_filtered=False,
                 **kwargs):
        super(Dota_Dataset, self).__init__(ann_file, pipeline, version=version,
                                               difficulty=difficulty, cache_annotations=cache_annotations,
                                               cache_filtered=cache_filtered, **kwargs)
  