from .dota import *
from .builder import ROTATED_DATASETS
from mmengine.dataset import BaseDataset
from typing import List, Optional

 
@ROTATED_DATASETS.register_module()
class SARDetDataset(DOTADataset):
 
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

    PALETTE = [(220, 120, 60),(220, 220, 60),(220, 20, 120),(220, 20, 220),(220, 20, 0),(220, 120, 0)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='le90',
                 difficulty=100,
                 cache_annotations=False,
                 cache_filtered=False,
                 **kwargs):
        super(SARDetDataset, self).__init__(ann_file, pipeline, version=version,
                                               difficulty=difficulty, cache_annotations=cache_annotations,
                                               cache_filtered=cache_filtered, **kwargs)
  