import numbers
from math import cos, pi
from typing import Callable, List, Optional, Union

import mmcv
from mmcv import runner
from mmcv.runner.hooks.hook import HOOKS, Hook
import torch
import numpy as np 
from scipy.special import kl_div
from math import sqrt
from mmcv.runner.hooks import LrUpdaterHook
import torch.nn.functional as F


def kl_divergence(p, q):
    # Ensure the inputs are numpy arrays
    p = p.numpy()
    q = q.numpy()
    
    # Normalize the lists to represent probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate KL divergence
    return np.sum(kl_div(p, q))


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
        return self.ema if self.ema is not None else 1e-3
    
@HOOKS.register_module()
class DynamicLrUpdaterHook(LrUpdaterHook):
    """Step LR scheduler with min_lr clipping.

    Args:
        step (int | list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (float): Decay LR ratio. Defaults to 0.1.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
    """

    def __init__(self,
                 step: Union[int, List[int]],
                 gamma: float = 0.1,
                 min_lr: Optional[float] = None,
                 extra_args = {'T':5, 'b':0.5, 'ema': 0.005, 'backbone_policy':'min', 'head_policy':'normal'},
                 reweight_losses={'sar_loss_cls':'sar_bbox_head','sar_loss_bbox':'sar_bbox_head','sar_loss_dfl':'sar_bbox_head',
                    'rgb_loss_rpn_cls':'rgb_rpn_head', 'rgb_loss_rpn_bbox':'rgb_rpn_head', 'rgb_loss_cls':'rgb_roi_head','rgb_loss_bbox':'rgb_roi_head',
                    'ifr_loss_rpn_cls':'ifr_rpn_head','ifr_loss_rpn_bbox':'ifr_rpn_head','ifr_loss_cls':'ifr_roi_head','ifr_loss_bbox':'ifr_roi_head'},
                 **kwargs) -> None:
        if isinstance(step, list):
            assert mmcv.is_list_of(step, int)
            assert all([s > 0 for s in step])
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        self.extra_args = extra_args
        self.reweight_losses = reweight_losses

        self.param_groups_param_names_mapping = {}
        self.reweight_losses = reweight_losses 
        self.T = extra_args['T']
        self.b = extra_args['b']
        self.history_ema_loss = [EMA_meter(extra_args['ema']) for _ in range(len(self.reweight_losses))]
        if self.extra_args['backbone_policy']=='sigmoid_kl':
            self.sigmoid = torch.nn.Sigmoid()
        super().__init__(**kwargs)

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        progress = runner.epoch if self.by_epoch else runner.iter
        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break
        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr

    def get_dynamic_lr(self, runner: 'runner.BaseRunner'):
        if hasattr(runner, "outputs"):
            losses = runner.outputs['log_vars'] 
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]
        cur_losses = {'name':[], 'loss':[]}
        for i, (k, loss) in enumerate(losses.items()):
            if k not in self.reweight_losses:
                continue
            elif isinstance(loss, list):
                loss = sum(loss)
            cur_losses['loss'].append(loss)
            cur_losses['name'].append(k)
        cur_losses['loss'] = torch.tensor(cur_losses['loss']) 
        num_losses = len(cur_losses['loss']) 
        if self.history_ema_loss[0].steps < self.warmup_iters or self.extra_args['head_policy']=='None': # warmup ema
            batch_weight = torch.ones(num_losses)
        else:
            history_loss = np.array([m.get() for m in self.history_ema_loss])
            if self.extra_args['head_policy']=='reverse':
                w_i = cur_losses['loss']/torch.tensor(history_loss)
            else:
                w_i = torch.tensor(history_loss)/cur_losses['loss']
            batch_weight = num_losses*torch.nn.functional.softmax(w_i/self.T, dim=-1)
            # if self.multi_tasks_reweight=='noisy_HDRS_loss':  
                # noise =((num_losses-1)/num_losses + torch.nn.functional.softmax(torch.randn(w_i.size()))) 
                # batch_weight = (num_losses*torch.nn.functional.softmax(w_i/self.T, dim=-1) + self.b)*noise
        subnet_lr_reweight = {k:1 for k in set(self.reweight_losses.values())}
        for subnet in subnet_lr_reweight.keys():
            lr_reweight = []
            for i, loss_name in enumerate(cur_losses['name']):
                if self.reweight_losses[loss_name] == subnet:
                    lr_reweight.append(batch_weight[i])
            lr_reweight = sum(lr_reweight)/len(lr_reweight)
            subnet_lr_reweight[subnet] = lr_reweight 

        new_lr = [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]
        if self.extra_args['backbone_policy']=='min':
            shared_lr_reweight = min(subnet_lr_reweight.values())
        elif self.extra_args['backbone_policy']=='avg':
            shared_lr_reweight = sum(subnet_lr_reweight.values())/len(subnet_lr_reweight.values())
        elif self.extra_args['backbone_policy']=='max':
            shared_lr_reweight = max(subnet_lr_reweight.values())
        elif self.extra_args['backbone_policy']=='kl':
            history_loss = torch.nn.functional.softmax(torch.tensor( np.array([m.get() for m in self.history_ema_loss])), dim=-1)
            cur_losses_ = torch.nn.functional.softmax(cur_losses['loss'], dim=-1)
            kl_div = F.kl_div(cur_losses_.log(), history_loss, reduction='batchmean')
            shared_lr_reweight = 1+ (1 - kl_div)/sqrt(self.T)
        elif self.extra_args['backbone_policy']=='sigmoid_kl':
            history_loss = torch.nn.functional.softmax(torch.tensor( np.array([m.get() for m in self.history_ema_loss])), dim=-1)
            cur_losses_ = torch.nn.functional.softmax(cur_losses['loss'], dim=-1)
            kl_div = F.kl_div(cur_losses_.log(), history_loss, reduction='batchmean')
            shared_lr_reweight = self.sigmoid((1-kl_div-self.b)* self.T)*2
        else:
            shared_lr_reweight = torch.tensor(1.0)

        for i, loss in enumerate(cur_losses['loss']):
            self.history_ema_loss[i].update(loss.item()) 
        # self.regular_lr = shared_lr_reweight
        for k,v in self.param_groups_param_names_mapping.items():
            is_shared = True
            for subnet, lr_reweight in subnet_lr_reweight.items():
                if subnet in v:
                    new_lr[k] = new_lr[k] *lr_reweight.item()
                    is_shared = False
                    break
            if is_shared:
                new_lr[k] = new_lr[k] * shared_lr_reweight.item()
        return new_lr

    def before_run(self, runner: 'runner.BaseRunner'):
        for idx, param_group in enumerate(runner.optimizer.param_groups):  # type: ignore
            for name, param in runner.model.named_parameters():
                assert len(param_group['params']) == 1
                param_group_ = param_group['params'][0] 

                if torch.equal(param_group_.data, param.data):
                    self.param_groups_param_names_mapping[idx] = name

        for group in runner.optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr']
            for group in runner.optimizer.param_groups  # type: ignore
        ]


    def before_train_iter(self, runner: 'runner.BaseRunner'):
        pass

    def after_train_iter(self, runner: 'runner.BaseRunner'):
        cur_iter = runner.iter
        assert isinstance(self.warmup_iters, int)
        if not self.by_epoch:
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self.dynamic_lr = self.get_dynamic_lr(runner)
                self._set_lr(runner, self.dynamic_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                if hasattr(runner, "outputs"):
                    losses = runner.outputs['log_vars'] 
                    cur_losses = {'loss':[]}
                    for i, (k, loss) in enumerate(losses.items()):
                        if k not in self.reweight_losses:
                            continue
                        elif isinstance(loss, list):
                            loss = sum(loss)
                        cur_losses['loss'].append(loss)
                    cur_losses['loss'] = torch.tensor(cur_losses['loss']) 
                    for i, loss in enumerate(cur_losses['loss']):
                        self.history_ema_loss[i].update(loss.item()) 
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            assert False