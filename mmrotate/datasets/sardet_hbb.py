from mmdet.datasets import CocoDataset 

from .builder import ROTATED_DATASETS
  
@ROTATED_DATASETS.register_module()
class SARDet_hbb(CocoDataset):

    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor',
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter')

    PALETTE = [(220, 120, 60),(220, 220, 60),(220, 20, 120),(220, 20, 220),(220, 20, 0),(220, 120, 0),
         (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100),  (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (0, 226, 252)]
