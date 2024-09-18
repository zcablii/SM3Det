# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from typing import Sequence
from functools import partial
from timm.models.layers import DropPath
from collections import OrderedDict

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.distributions.normal import Normal



from mmengine.model import ModuleList, Sequential
from mmengine.logging import MMLogger 
from mmengine.runner.checkpoint import CheckpointLoader

from mmcv.cnn import (build_activation_layer,
                      constant_init, trunc_normal_init)
import torch
import torch.nn as nn
from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, trunc_normal_


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]
    def forward(self, x, data_format='channel_first'):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            # If the output is discontiguous, it may cause some unexpected
            # problem in the downstream tasks
            x = x.permute(0, 3, 1, 2).contiguous()
        return x
def build_LayerNorm2d_layer(cfg: dict, num_features: int) -> nn.Module:
 
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    norm_layer = LayerNorm2d
    # if norm_layer is None:
    #     raise KeyError(f'Cannot find {layer_type} in registry under scope '
    #                    f'name {MODELS.scope}')
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    layer = norm_layer(num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer
class GRN(nn.Module):
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor, data_format='channel_first'):
        if data_format == 'channel_last':
            gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
            x = self.gamma * (x * nx) + self.beta + x
        elif data_format == 'channel_first':
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
            x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(
                1, -1, 1, 1) + x
        return x

class CosineTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, init_t=0.5):
        super(CosineTopKGate, self).__init__()
        proj_dim = min(model_dim//2, 256)
        # proj_dim=256
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = torch.nn.Linear(model_dim, proj_dim)
        self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x):
        cosine_projector = self.cosine_projector
        sim_matrix = self.sim_matrix
        logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
                              F.normalize(sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits
        
class MoE_layer(nn.Module):
    def __init__(self, 
                 moe_cfg
                 ):
        super(MoE_layer, self).__init__() 
        self.noisy_gating = moe_cfg['noisy_gating']
        self.num_experts = moe_cfg['num_experts']
        self.input_size = moe_cfg['in_channels']
        self.k = moe_cfg['top_k']
        # instantiate experts
        self.gating = moe_cfg['gating']
        self.experts = nn.ModuleList([FFN(
            in_channels=self.input_size,
            mid_channels=moe_cfg['mid_channels'], 
            pw_conv=moe_cfg['pw_conv'],
            act_cfg=moe_cfg['act_cfg'],
            use_grn=moe_cfg['use_grn']) for i in range(self.num_experts)]) 
        self.infer_expert = None
 
        if moe_cfg['gating'] == 'linear':
            self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
        elif moe_cfg['gating'] == 'cosine':
            self.w_gate = CosineTopKGate(self.input_size, self.num_experts)
        self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        if len(x.shape) == 2:
            x = x.sum(dim=0)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten() # (bs x m)
        threshold_positions_if_in = torch.arange(batch) * m + self.k # bs
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in.to(top_values_flat.device)), 1)

        if len(noisy_values.shape) == 3:
            threshold_if_in = threshold_if_in.unsqueeze(1)

        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out.to(top_values_flat.device)), 1)
        if len(noisy_values.shape) == 3:
            threshold_if_out = threshold_if_out.unsqueeze(1)

        # is each value currently in the top k.

        normal = Normal(self.mean.to(noise_stddev.device), self.std.to(noise_stddev.device))
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    
    def random_k_gating(self, features, train):
        if train:
            idx = torch.randint(0, self.num_experts, 1)
            results = self.experts[idx](features)

        else:
            results = []
            for i in range(self.num_experts):
                tmp = self.num_experts[i](features)
                results.append(tmp)
            
            results = torch.stack(results, dim=0).mean(dim=0)

        return results



    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        if self.gating == 'linear':
            clean_logits = x @ self.w_gate
        elif self.gating == 'cosine':
            clean_logits = self.w_gate(x)

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim= -1)  
        
        top_k_logits = top_logits[:, :self.k] if len(top_logits.shape) == 2 else top_logits[:, :, :self.k]    
        top_k_indices = top_indices[:, :self.k] if len(top_indices.shape) == 2 else top_indices[:, :, :self.k]
        
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
       
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)  

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


    def forward(self, x, loss_coef=1e-2):
        train = self.training 
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_shape = x.shape 
        x = x.reshape(-1,x.shape[-1])
        gates, load = self.noisy_top_k_gating(x, train)
        importance = gates.sum(dim=0)
        
        # calculate loss
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        
        expert_inputs = dispatcher.dispatch(x) 
        gates = dispatcher.expert_to_gates() 
        expert_outputs = [self.experts[i](expert_inputs[i]).reshape(-1, x_shape[-1]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(x_shape)
        # assert False, (y.shape, y[0][0][0])
        return y, loss

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)  # torch.nonzero: 
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)


    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            if len(stitched.shape) == 3:
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(1))
            else:
                stitched = stitched.mul(self._nonzero_gates)

        if len(stitched.shape) == 3:
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(-1), requires_grad=True, device=stitched.device)
        else:
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class DALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )] * 3)
        self.dataset_DA = {'sar':0,'rgb':1,'ifr':2}

    def forward(self, x, dataset):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        if len(dataset) == 1: 
            y = self.fc[self.dataset_DA[dataset[0]]](y).view(b, c, 1, 1) 
            return x * y.expand_as(x)
        else:
            ys = []
            for x_, dataset_ in zip(y, dataset):
                y_ = self.fc[self.dataset_DA[dataset_]](x_.view(1, c))
                ys.append(y_)
            y = torch.cat(ys, dim=1).view(b, c, 1, 1) 
            return x * y.expand_as(x)


class ConvNeXtBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 MoE_cfg=None,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)
        self.linear_pw_conv = linear_pw_conv
        self.norm = build_LayerNorm2d_layer(norm_cfg, in_channels)
        mid_channels = int(mlp_ratio * in_channels) 
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)
        self.ffn = FFN(in_channels,mid_channels,pw_conv,act_cfg,use_grn)

        self.MoE_cfg = MoE_cfg
        if MoE_cfg is not None:
            MoE_cfg.update({'in_channels': in_channels, 
                'mid_channels': mid_channels, 
                'pw_conv': pw_conv,'use_grn': use_grn, 
                'act_cfg':act_cfg})
            self.ffn = MoE_layer(MoE_cfg)
        else:
            self.ffn = FFN(in_channels,mid_channels,pw_conv,act_cfg,use_grn)


        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.DA = DALayer(in_channels)


    def forward(self, x, dataset):
        
        def _inner_forward(x):
            shortcut = x
            loss = None
            # print(x)
            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                if self.MoE_cfg is not None:
                    
                    x,loss = self.ffn(x)
                else: 
                    x = self.ffn(x)
                
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                if self.MoE_cfg is not None:
                    
                    x,loss = self.ffn(x)
                else: 
                    x = self.ffn(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))
            x = self.drop_path(self.DA(x, dataset))
            x = shortcut + x
             
            return x, loss

        if self.with_cp and x.requires_grad:
            x, loss = cp.checkpoint(_inner_forward, x)
        else:
            x, loss = _inner_forward(x)
        
        return x, loss

class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 pw_conv,
                 act_cfg=dict(type='GELU'),
                 use_grn=False):
        super().__init__()
        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        if use_grn:
            self.grn = GRN(mid_channels)
        else:
            self.grn = None
    def forward(self, x):
        x = self.pointwise_conv1(x)
        # print(x)
        x = self.act(x)
        if self.grn is not None:
            x = self.grn(x, data_format='channel_last')
         
        x = self.pointwise_conv2(x) 
        return x

 
class ConvNeXt_moe(BaseModule):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'swin_large': {
            'depths': [2,  2, 18,  2],
            'channels': [192, 384, 768, 1536]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 MoE_Block_inds = [[],[],[],[]],
                 noisy_gating= True, 
                 num_experts= 2, 
                 gate= 'cosine', 
                 top_k= 2,
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.MoE_Block_inds = MoE_Block_inds
        self.num_experts = num_experts
        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_LayerNorm2d_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_LayerNorm2d_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)
            MoE_Block_ind = [list(range(depth))[q] for q in self.MoE_Block_inds[i] if q < depth]
            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    MoE_cfg = {'noisy_gating': noisy_gating, 'num_experts': num_experts, 'top_k': top_k, 'gating': gate} if j in MoE_Block_ind else None,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_LayerNorm2d_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def forward(self, x):
        outs = []
        gate_losses = []
        for i, stage in enumerate(self.stages): 
            x = self.downsample_layers[i](x)
            for each_layer in stage:
                x, gate_loss = each_layer(x)
                if gate_loss is not None:
                    gate_losses.append(gate_loss)  
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x))
        if len(gate_losses) > 0:
            return tuple(outs), sum(gate_losses)/len(gate_losses)
        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_moe, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2

    def init_weights(self): 
        def add_experts_inits():

            pass
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
                
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    k = k[9:]
                    if 'pointwise_conv' in k:
                        stage_splits = k.split('.') 
                        stage_ind = eval(stage_splits[1])
                        blocks_ind = eval(stage_splits[2]) 
                        # if blocks_ind in [list(range(self.depth))[q] for q in self.MoE_Block_inds[stage_ind] if q < self.depth]:
                        if blocks_ind in self.MoE_Block_inds[stage_ind]:
                            for expert_idx in range(self.num_experts):
                                new_k = k.replace('pointwise_conv', 'ffn.experts.'+str(expert_idx)+'.pointwise_conv')
                                state_dict[new_k] = v
                        else:
                            new_k = k.replace('pointwise_conv', 'ffn.pointwise_conv') 
                            state_dict[new_k] = v
                    elif 'grn' in k:
                        
                        stage_splits = k.split('.') 
                        stage_ind = eval(stage_splits[1])
                        blocks_ind = eval(stage_splits[2]) 
                        if blocks_ind in self.MoE_Block_inds[stage_ind]:
                            for expert_idx in range(self.num_experts):
                                new_k = k.replace('grn', 'ffn.experts.'+str(expert_idx)+'.grn')
                                state_dict[new_k] = v
                        else:        
                            new_k = k.replace('grn', 'ffn.grn')
                            state_dict[new_k] = v
                    else:
                        state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict 
            warnings = self.load_state_dict(state_dict, False)
            print(warnings)


@ROTATED_BACKBONES.register_module()
class ConvNeXt_DA_MultiInput(ConvNeXt_moe):
    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 datasets=None,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 MoE_Block_inds = [[],[],[],[]],
                 noisy_gating= True, 
                 num_experts= 2, 
                 top_k= 2,
                 gate= 'cosine', 
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(
                 MoE_Block_inds = MoE_Block_inds,
                 noisy_gating= noisy_gating, 
                 num_experts= num_experts, 
                 gate= gate, 
                 top_k= top_k, 
                 arch=arch,
                 in_channels=in_channels,
                 stem_patch_size=stem_patch_size,
                 norm_cfg=norm_cfg,
                 act_cfg=act_cfg,
                 linear_pw_conv=linear_pw_conv,
                 use_grn=use_grn,
                 drop_path_rate=drop_path_rate,
                 layer_scale_init_value=layer_scale_init_value,
                 out_indices=out_indices,
                 frozen_stages=frozen_stages,
                 gap_before_final_norm=gap_before_final_norm,
                 with_cp=with_cp,init_cfg=init_cfg)
  
        self.downsample_layers[0] = nn.Sequential(build_LayerNorm2d_layer(norm_cfg, self.channels[0]))
        
        self.init_datasets = datasets
        self.datasets = datasets if datasets is not None else ['single']
         

        self.dataset_stems = nn.ModuleDict()
        for dataset in self.datasets:
            self.dataset_stems[dataset]=nn.Conv2d(
                                            in_channels,
                                            self.channels[0],
                                            kernel_size=stem_patch_size,
                                            stride=stem_patch_size)
                
    
    def forward(self, x, datasets=['single']):
        outs = []
        # assert len(x) == self.datasets
        batch_input = []
        if len(datasets) == 1:
            x = [x]
        
        x = torch.cat(x,dim=0)
        batch_input = [self.dataset_stems['single'](x)]
        
        x = torch.cat(batch_input, dim=0)
        gate_losses = []
        for i, stage in enumerate(self.stages): 
            x = self.downsample_layers[i](x)
            for each_layer in stage:
                x, gate_loss = each_layer(x, datasets)
                if gate_loss is not None:
                    gate_losses.append(gate_loss)  
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x))
        if len(gate_losses) > 0:
            return tuple(outs), sum(gate_losses)/len(gate_losses)
        return tuple(outs)
 


    def init_weights(self): 
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
                
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    k = k[9:]
                    
                    if 'downsample_layers.0.0' in k:
                        for i in self.datasets:
                            new_k = k.replace('downsample_layers.0.0', 'dataset_stems.'+str(i))
                            state_dict[new_k] = v 
                    elif 'downsample_layers.0.1' in k:
                        for i in self.datasets:
                            new_k = k.replace('downsample_layers.0.1', 'downsample_layers.0.0')
                            state_dict[new_k] = v


                    elif 'pointwise_conv' in k:
                        stage_splits = k.split('.') 
                        stage_ind = eval(stage_splits[1])
                        blocks_ind = eval(stage_splits[2]) 
                        # if blocks_ind in [list(range(self.depth))[q] for q in self.MoE_Block_inds[stage_ind] if q < self.depth]:
                        if blocks_ind in self.MoE_Block_inds[stage_ind]:
                            for expert_idx in range(self.num_experts):
                                new_k = k.replace('pointwise_conv', 'ffn.experts.'+str(expert_idx)+'.pointwise_conv')
                                state_dict[new_k] = v
                        else:
                            new_k = k.replace('pointwise_conv', 'ffn.pointwise_conv') 
                            state_dict[new_k] = v
                    elif 'grn' in k:
                        
                        stage_splits = k.split('.') 
                        stage_ind = eval(stage_splits[1])
                        blocks_ind = eval(stage_splits[2]) 
                        if blocks_ind in self.MoE_Block_inds[stage_ind]:
                            for expert_idx in range(self.num_experts):
                                new_k = k.replace('grn', 'ffn.experts.'+str(expert_idx)+'.grn')
                                state_dict[new_k] = v
                        else:        
                            new_k = k.replace('grn', 'ffn.grn')
                            state_dict[new_k] = v
                    else:
                        state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict 
            warnings = self.load_state_dict(state_dict, False)
            print(warnings)
 

            for i, stage in enumerate(self.stages): 
                for each_layer in stage:
                    nn.init.constant_(each_layer.DA.fc[0][2].weight, 0.0)
                    nn.init.constant_(each_layer.DA.fc[1][2].weight, 0.0)
                    nn.init.constant_(each_layer.DA.fc[2][2].weight, 0.0)
                    


