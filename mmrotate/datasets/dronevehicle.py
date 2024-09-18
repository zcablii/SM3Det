from .dota import *
from .builder import ROTATED_DATASETS
from mmengine.dataset import BaseDataset
from typing import List, Optional

 
@ROTATED_DATASETS.register_module()
class DroneVehicle_Dataset(DOTADataset):
 
    CLASSES = ('CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN')

    PALETTE = [(255, 128, 0), (255, 0, 255), (0, 255, 255), (255, 193, 193), (0, 51, 153)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='le90',
                 difficulty=100,
                 cache_annotations=False,
                 cache_filtered=False,
                 **kwargs):
        super(DroneVehicle_Dataset, self).__init__(ann_file, pipeline, version=version,
                                               difficulty=difficulty, cache_annotations=cache_annotations,
                                               cache_filtered=cache_filtered, **kwargs)
  