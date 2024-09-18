# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmrotate.core import rbbox2result
import torch
from mmdet.core import bbox2result
import numpy as np
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector 
from .utils import AlignConvModule
 
class EMA_meter:
    def __init__(self, beta):
        self.beta = beta
        self.ema = None
        self.steps = 0

    def update(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = (1 - self.beta) * self.ema + self.beta * value
        self.steps += 1
    
    def get(self):
        return self.ema

@ROTATED_DETECTORS.register_module()
class TriSourceTwoOneDetector(RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """ 

    def __init__(self,
                 backbone,
                 neck=None,
                 rgb_fam_head=None,
                 rgb_align_cfgs=None,
                 rgb_odm_head=None,
                 rgb_train_cfg=None,
                 rgb_test_cfg=None,
                 ifr_fam_head=None,
                 ifr_align_cfgs=None,
                 ifr_odm_head=None,
                 ifr_train_cfg=None,
                 ifr_test_cfg=None,
                 sar_rpn_head=None,
                 sar_roi_head=None,
                 sar_train_cfg=None,
                 sar_test_cfg=None,
                 multi_tasks_reweight=None,
                 reweight_losses=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TriSourceTwoOneDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.train_datasets = ['sar', 'rgb', 'ifr']

        if neck is not None:
            self.neck = build_neck(neck)


        if rgb_train_cfg is not None:
            rgb_fam_head.update(train_cfg=rgb_train_cfg['fam_cfg'])
        rgb_fam_head.update(test_cfg=rgb_test_cfg)
        self.rgb_fam_head = build_head(rgb_fam_head)

        self.align_conv_type = rgb_align_cfgs['type']
        self.align_conv_size = rgb_align_cfgs['kernel_size']
        self.feat_channels = rgb_align_cfgs['channels']
        self.featmap_strides = rgb_align_cfgs['featmap_strides']

        if self.align_conv_type == 'AlignConv':
            self.rgb_align_conv = AlignConvModule(self.feat_channels,
                                              self.featmap_strides,
                                              self.align_conv_size)

        if rgb_train_cfg is not None:
            rgb_odm_head.update(train_cfg=rgb_train_cfg['odm_cfg'])
        rgb_odm_head.update(test_cfg=rgb_test_cfg)
        self.rgb_odm_head = build_head(rgb_odm_head)

        self.rgb_train_cfg = rgb_train_cfg
        self.rgb_test_cfg = rgb_test_cfg

        #######################################################
        if ifr_train_cfg is not None:
            ifr_fam_head.update(train_cfg=ifr_train_cfg['fam_cfg'])
        ifr_fam_head.update(test_cfg=ifr_test_cfg)
        self.ifr_fam_head = build_head(ifr_fam_head)

        self.align_conv_type = ifr_align_cfgs['type']
        self.align_conv_size = ifr_align_cfgs['kernel_size']
        self.feat_channels = ifr_align_cfgs['channels']
        self.featmap_strides = ifr_align_cfgs['featmap_strides']

        if self.align_conv_type == 'AlignConv':
            self.ifr_align_conv = AlignConvModule(self.feat_channels,
                                              self.featmap_strides,
                                              self.align_conv_size)

        if ifr_train_cfg is not None:
            ifr_odm_head.update(train_cfg=ifr_train_cfg['odm_cfg'])
        ifr_odm_head.update(test_cfg=ifr_test_cfg)
        self.ifr_odm_head = build_head(ifr_odm_head)

        self.ifr_train_cfg = ifr_train_cfg
        self.ifr_test_cfg = ifr_test_cfg


        #################################################### HBB
        if sar_rpn_head is not None:
            sar_rpn_train_cfg = sar_train_cfg.rpn if sar_train_cfg is not None else None
            sar_rpn_head_ = sar_rpn_head.copy()
            sar_rpn_head_.update(train_cfg=sar_rpn_train_cfg, test_cfg=sar_test_cfg.rpn)
            self.sar_rpn_head = build_head(sar_rpn_head_)

        if sar_roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            sar_rcnn_train_cfg = sar_train_cfg.rcnn if sar_train_cfg is not None else None
            sar_roi_head.update(train_cfg=sar_rcnn_train_cfg)
            sar_roi_head.update(test_cfg=sar_test_cfg.rcnn)
            sar_roi_head.pretrained = pretrained
            self.sar_roi_head = build_head(sar_roi_head)

        self.sar_train_cfg = sar_train_cfg
        self.sar_test_cfg = sar_test_cfg


        self.multi_tasks_reweight = multi_tasks_reweight
        self.reweight_losses = reweight_losses
        if self.multi_tasks_reweight=='uncertainty':
            task_num = len(self.reweight_losses) 
            self.mtl_sigma = torch.nn.Parameter(torch.ones(task_num, requires_grad=True))
        elif self.multi_tasks_reweight=='dwa':
            self.T = 2
            self.history_loss = None

    @property
    def with_sar_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'sar_rpn_head') and self.sar_rpn_head is not None

    
    @property
    def with_sar_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'sar_roi_head') and self.sar_roi_head is not None


    def extract_feat(self, batch_inputs, datasets, is_train=False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(batch_inputs, datasets)
        if isinstance(x, tuple) and len(x) ==2:
            x, loss = x
            experts_id=None
        ###
        elif isinstance(x, tuple) and len(x) ==3:
            x,loss,experts_id=x
        ###
        else :
            loss = None
            experts_id=None
        if self.with_neck:
            if len(datasets)>1:
                assert is_train
                sar_x, rgb_x, ifr_x = self.split_batch(x)
                sar_x = self.neck(sar_x)
                rgb_x = self.neck(rgb_x, start_level=1, add_extra_convs='on_output')
                ifr_x = self.neck(ifr_x, start_level=1, add_extra_convs='on_output')
                x = (sar_x, rgb_x, ifr_x)
            else:
                assert not is_train
                if datasets[0] == 'sar':
                    x = self.neck(x)
                elif datasets[0] in ['rgb', 'ifr']:
                    x = self.neck(x, start_level=1, add_extra_convs='on_output')
                else:
                    assert False, 'Invalid dataset'

        if is_train:
            return x, loss
        return x,experts_id
    
    def split_batch(self, x, is_list=False): 
        if is_list:
            slices = []
            start = 0
            for length in self.source_ratio:
                end = start + length
                slices.append(x[start:end])
                start = end

            return slices
        else:
            slices = [torch.split(x_, self.source_ratio, dim=0) for x_ in x]
            return tuple(map(list, zip(*slices))) 
        
   
    def gather_dict_values(self, data, ignore_tensor=False):
        gathered_info = {namespace: [] for namespace in self.train_datasets}

        for item in data:
            for namespace in self.train_datasets:
                if item.get(namespace) is not None:
                    gathered_info[namespace].append(item[namespace])

        for namespace in self.train_datasets:
            if gathered_info[namespace] and isinstance(gathered_info[namespace][0], torch.Tensor):
                # Move each tensor to CUDA
                if ignore_tensor:
                    gathered_info[namespace] = [tensor.cuda() for tensor in gathered_info[namespace]]
                else:                
                    gathered_info[namespace] = torch.stack(gathered_info[namespace]).cuda()

        return gathered_info
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        # assert False, (img,img_metas)
        # print(img_metas)
        # print('==+'*10) 
        assert gt_bboxes_ignore is None
        img = self.gather_dict_values(img) 
        img_metas = self.gather_dict_values(img_metas)
        gt_bboxes = self.gather_dict_values(gt_bboxes, ignore_tensor=True)
        gt_labels = self.gather_dict_values(gt_labels, ignore_tensor=True)
 
        sar_img_metas = img_metas['sar'] 
        rgb_img_metas = img_metas['rgb']
        ifr_img_metas = img_metas['ifr']

        sar_gt_bboxes = gt_bboxes['sar']
        rgb_gt_bboxes = gt_bboxes['rgb']
        ifr_gt_bboxes = gt_bboxes['ifr']

        sar_gt_labels = gt_labels['sar']
        rgb_gt_labels = gt_labels['rgb']
        ifr_gt_labels = gt_labels['ifr']
        
        self.source_ratio = [len(sar_gt_labels), len(rgb_gt_labels), len(ifr_gt_labels)]


        batch_inputs = []  
        # for each in self.train_datasets:
        #     batch_inputs.append(img[each])
        for each in self.train_datasets:
            if len(img[each])>0:
                batch_inputs.append(img[each])


        x, gate_loss = self.extract_feat(batch_inputs, self.train_datasets, is_train=True)
        losses = dict()
        if gate_loss is not None:
            losses.update({'gate_loss': gate_loss})
 
        sar_x, rgb_x, ifr_x = x
        
        
        if len(sar_gt_labels) > 0:
            if self.with_sar_rpn:
                proposal_cfg = self.sar_train_cfg.get('rpn_proposal',
                                                self.sar_test_cfg.rpn)
                rpn_losses, proposal_list = self.sar_rpn_head.forward_train(
                    sar_x,
                    sar_img_metas,
                    sar_gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
                losses.update({'sar_' + k: v for k, v in rpn_losses.items()})
            else:
                proposal_list = proposals

            roi_losses = self.sar_roi_head.forward_train(sar_x, sar_img_metas, proposal_list,
                                                    sar_gt_bboxes, sar_gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            losses.update({'sar_' + k: v for k, v in roi_losses.items()})
            
###############################################################
        rgb_outs = self.rgb_fam_head(rgb_x)

        rgb_loss_inputs = rgb_outs + (rgb_gt_bboxes, rgb_gt_labels, rgb_img_metas)
        rgb_loss_base = self.rgb_fam_head.loss(
            *rgb_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in rgb_loss_base.items():
            losses[f'rgb_fam.{name}'] = value

        rois = self.rgb_fam_head.refine_bboxes(*rgb_outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.rgb_align_conv(rgb_x, rois)
        rgb_outs = self.rgb_odm_head(align_feat)
        rgb_loss_inputs = rgb_outs + (rgb_gt_bboxes, rgb_gt_labels, rgb_img_metas)
        loss_refine = self.rgb_odm_head.loss(
            *rgb_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
        for name, value in loss_refine.items():
            losses[f'rgb_odm.{name}'] = value
########################################################

        ifr_outs = self.ifr_fam_head(ifr_x)

        ifr_loss_inputs = ifr_outs + (ifr_gt_bboxes, ifr_gt_labels, ifr_img_metas)
        ifr_loss_base = self.ifr_fam_head.loss(
            *ifr_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in ifr_loss_base.items():
            losses[f'ifr_fam.{name}'] = value

        rois = self.ifr_fam_head.refine_bboxes(*ifr_outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.ifr_align_conv(ifr_x, rois)
        ifr_outs = self.ifr_odm_head(align_feat)
        ifr_loss_inputs = ifr_outs + (ifr_gt_bboxes, ifr_gt_labels, ifr_img_metas)
        loss_refine = self.ifr_odm_head.loss(
            *ifr_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
        for name, value in loss_refine.items():
            losses[f'ifr_odm.{name}'] = value

        loss_sum = 0 
        if self.multi_tasks_reweight is not None:
            reweight_losses = {}
            cur_losses = []
            for i, (k, loss) in enumerate(losses.items()):
                if k not in self.reweight_losses:
                    reweight_losses.update({k:loss})
                    continue
                elif isinstance(loss, list):
                    loss = sum(loss)
                cur_losses.append(loss)
            cur_losses = torch.stack(cur_losses) 

        if self.multi_tasks_reweight=='uncertainty':
            for i, loss in enumerate(cur_losses):
                loss_sum += 0.5 / (self.mtl_sigma[i] ** 2) * loss + torch.log(1 + self.mtl_sigma[i] ** 2)   
            reweight_losses.update({'reweighted_total_losses':loss_sum})
            return reweight_losses
        
        elif self.multi_tasks_reweight=='dwa':   
            
            if self.history_loss is not None:
                w_i = cur_losses/torch.tensor(self.history_loss).cuda()
                batch_weight = len(self.reweight_losses)*torch.nn.functional.softmax(w_i/self.T, dim=-1)
            else:
                batch_weight = torch.ones_like(losses).cuda()
            loss_sum = torch.mul(cur_losses, batch_weight).sum()
        
            reweight_losses.update({'reweighted_total_losses':loss_sum})
            self.history_loss = cur_losses.detach().cpu().numpy()
            
            return reweight_losses
         
     
        
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                subdataset,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation.""" 
        assert False, 'Not implemented yet'
        assert isinstance(subdataset[0],list) and len(subdataset)==1
        assert all(x == subdataset[0][0] for x in subdataset[0]), "Not all elements in subdataset are the same: " + str(subdataset)
        subdataset = subdataset[0][0]
        x = self.extract_feat(img, [subdataset]) 
        if subdataset == 'sar':
            rpn_head = self.sar_rpn_head
            roi_head = self.sar_roi_head
        elif subdataset == 'rgb':
            rpn_head = self.rgb_rpn_head
            roi_head = self.rgb_roi_head 

        if proposals is None:
            proposal_list = await rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, subdataset, proposals=None, rescale=False):
        """Test without augmentation."""
        assert isinstance(subdataset[0],list) and len(subdataset)==1 # subdataset: [['sar']]
        assert all(x == subdataset[0][0] for x in subdataset[0]), "Not all elements in subdataset are the same: " + str(subdataset)
        subdataset = subdataset[0][0]
        # print(img.shape)
        #expert_id
        x = self.extract_feat(img, [subdataset]) 
        # for i in x:
        #     print(i.shape)
        if isinstance(x,tuple):
            x,experts_id=x
        else:
            experts_id=None
        if subdataset == 'sar':
            if proposals is None:
                 proposal_list = self.sar_rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            return self.sar_roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)
        
        elif subdataset == 'rgb':
            outs = self.rgb_fam_head(x)
            rois = self.rgb_fam_head.refine_bboxes(*outs)
            # rois: list(indexed by images) of list(indexed by levels)
            align_feat = self.rgb_align_conv(x, rois)
            outs = self.rgb_odm_head(align_feat)

            bbox_inputs = outs + (img_metas, self.rgb_test_cfg, rescale)
            bbox_list = self.rgb_odm_head.get_bboxes(*bbox_inputs, rois=rois)
            bbox_results = [
                rbbox2result(det_bboxes, det_labels, self.rgb_odm_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results
        

        elif subdataset == 'ifr':
            outs = self.ifr_fam_head(x)
            rois = self.ifr_fam_head.refine_bboxes(*outs)
            # rois: list(indexed by images) of list(indexed by levels)
            align_feat = self.ifr_align_conv(x, rois)
            outs = self.ifr_odm_head(align_feat)

            bbox_inputs = outs + (img_metas, self.ifr_test_cfg, rescale)
            bbox_list = self.ifr_odm_head.get_bboxes(*bbox_inputs, rois=rois)
            bbox_results = [
                rbbox2result(det_bboxes, det_labels, self.ifr_odm_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results

    def aug_test(self, imgs, img_metas,subdataset, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        assert isinstance(subdataset[0],list) and len(subdataset)==1
        assert all(x == subdataset[0][0] for x in subdataset[0]), "Not all elements in subdataset are the same: " + str(subdataset)
        subdataset = subdataset[0][0]
        x = self.extract_feat(imgs, [subdataset]) 
        if subdataset == 'sar':
            results_list = self.sar_bbox_head.aug_test(
            x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results
        elif subdataset == 'rgb':
            proposal_list = self.rgb_rpn_head.aug_test_rpn(x, img_metas)
            return self.rgb_roi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale)

        elif subdataset == 'ifr':
            proposal_list = self.ifr_rpn_head.aug_test_rpn(x, img_metas)
            return self.ifr_roi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale)