import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer
from mmengine.model import ModuleList, Sequential
from mmengine.logging import MMLogger 
from mmengine.runner.checkpoint import CheckpointLoader
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.distributions.normal import Normal
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
    requires_grad = cfg_.pop('requires_grad', True)#default is true
    cfg_.setdefault('eps', 1e-5)
    layer = norm_layer(num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer
class CosineTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, init_t=0.5):
        super(CosineTopKGate, self).__init__()
        proj_dim = min(model_dim//2, 256)
        # proj_dim=256
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)#log2
        self.cosine_projector = torch.nn.Linear(model_dim, proj_dim)
        self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()#log100
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x):
        cosine_projector = self.cosine_projector
        sim_matrix = self.sim_matrix
        logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
                              F.normalize(sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()#限制上界
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
        self.output_size=moe_cfg['out_channels']
        self.k = moe_cfg['top_k']
        # instantiate experts
        self.gating = moe_cfg['gating']
        self.experts = nn.ModuleList([nn.Conv2d(self.input_size,self.output_size,1) for i in range(self.num_experts)]) 
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
        return x.float().var() / (x.float().mean()**2 + eps)#D(x)/(E(X)^2)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten() # (bs x m)
        threshold_positions_if_in = torch.arange(batch) * m + self.k # bs    #(0+k,m+k,2*m,···，batch*m+k)
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in.to(top_values_flat.device)), 1)#get all the k-th and reshape to[ [],[],[] ]

        if len(noisy_values.shape) == 3:
            threshold_if_in = threshold_if_in.unsqueeze(1)

        is_in = torch.gt(noisy_values, threshold_if_in)#get best k only 0 or 1
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out.to(top_values_flat.device)), 1)#(0+k-1,m+k-1,...,batch*m+k-1)
        if len(noisy_values.shape) == 3:
            threshold_if_out = threshold_if_out.unsqueeze(1)

        # is each value currently in the top k.

        normal = Normal(self.mean.to(noise_stddev.device), self.std.to(noise_stddev.device))
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)#P
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
            clean_logits = x @ self.w_gate#matmul
        elif self.gating == 'cosine':
            clean_logits = self.w_gate(x)

        if self.noisy_gating and train:#give noise
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
        #z=self.experts(x)
        x = x.permute(0, 2, 3, 1)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_shape = x.shape 
        x = x.reshape(-1,x.shape[-1])
        
        gates, load = self.noisy_top_k_gating(x, train)
        importance = gates.sum(dim=0)#每个专家的重要性
        
        # calculate loss
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x) 
        gates = dispatcher.expert_to_gates()
        expert_outputs=[] 
        for i in range(self.num_experts):
            if expert_inputs[i].shape[0] != 0:
                expert_outputs.append(self.experts[i](expert_inputs[i].reshape(-1,x_shape[-1],1,1)).reshape(expert_inputs[i].shape[0],-1))
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(x_shape[0],x_shape[1],x_shape[2],-1)
        y = y.permute(0, 3, 1, 2).contiguous()
        #print(y[0][0][1])
        #print(z[0][0][1])
        #print(y[0][1][2][0])
        #print(z[0][1][2][0])
#        assert y.data==z.data
        # assert False, (y.shape, y[0][0][0])
        
        return y,loss

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)  # torch.nonzero: index_sorted_experts:
        _, self._expert_index = sorted_experts.split(1, dim=1)#
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
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,MoE_cfg1=None,MoE_cfg2=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.MoE_cfg1=MoE_cfg1
        self.MoE_cfg2=MoE_cfg2
        if MoE_cfg1 is not None:
            MoE_cfg1.update({'in_channels': in_features, 
                'out_channels': hidden_features, 
                })
            self.fc1 = MoE_layer(MoE_cfg1)
        else:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        if MoE_cfg2 is not None:
            MoE_cfg2.update({'in_channels': hidden_features, 
                'out_channels': out_features, 
                })
            self.fc2 = MoE_layer(MoE_cfg2) 
        else: 
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        loss=[]
        if self.MoE_cfg1 is not None:
            x,loss1 = self.fc1(x)
            loss.append(loss1)
        else:
            x=self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        if self.MoE_cfg2 is not None:
            x,loss2 = self.fc2(x)
            loss.append(loss2)
        else :
            x=self.fc2(x)
        x = self.drop(x)
        if len(loss) > 0:
            return x, sum(loss)/len(loss)
        return x
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, norm_cfg=None,MoE_cfg1=None,MoE_cfg2=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.MoE_cfg1 = MoE_cfg1
        self.MoE_cfg2 = MoE_cfg2
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,MoE_cfg1=MoE_cfg1,MoE_cfg2=MoE_cfg2)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        loss=None
        if self.MoE_cfg1 is not None or self.MoE_cfg2 is not None:
            x_mem=x
            x,loss= self.mlp(self.norm2(x))
            x = x_mem + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *x)
        else:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x,loss


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

@ROTATED_BACKBONES.register_module()
class VAN_moe(BaseModule):
    def __init__(self, MoE_Block_inds_fc1 = [[],[],[],[]],MoE_Block_inds_fc2 = [[],[],[],[]],num_experts=2,top_k=2,img_size=224,noisy_gating= False,gate= 'cosine', 
                  in_chans=3, embed_dims=[32, 64, 160, 256],
                mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 3, 5, 2], num_stages=4, 
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.embed_dims=embed_dims
        self.num_stages = num_stages
        self.num_experts = num_experts
        self.MoE_Block_inds_fc1 = MoE_Block_inds_fc1
        self.MoE_Block_inds_fc2 = MoE_Block_inds_fc2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            depth = self.depths[i]
            MoE_Block_ind_fc1 = [list(range(depth))[q] for q in self.MoE_Block_inds_fc1[i] if q < depth]
            MoE_Block_ind_fc2 = [list(range(depth))[q] for q in self.MoE_Block_inds_fc2[i] if q < depth]
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j],norm_cfg=norm_cfg,
                MoE_cfg1={'noisy_gating': noisy_gating, 'num_experts': num_experts, 'top_k': top_k, 'gating': gate} if j in MoE_Block_ind_fc1 else None,
                MoE_cfg2={'noisy_gating': noisy_gating, 'num_experts': num_experts, 'top_k': top_k, 'gating': gate} if j in MoE_Block_ind_fc2 else None)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)



    def init_weights(self):
        logger = MMLogger.get_current_instance()
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('block'):
                    if 'fc' in k:
                        k = k[5:]
                        stage_splits = k.split('.') 
                        stage_ind = eval(stage_splits[0])-1
                        blocks_ind = eval(stage_splits[1]) 
                        # if blocks_ind in [list(range(self.depth))[q] for q in self.MoE_Block_inds[stage_ind] if q < self.depth]:
                        k='block'+k
                        if 'fc1' in k:
                            if blocks_ind in self.MoE_Block_inds_fc1[stage_ind]:
                                for expert_idx in range(self.num_experts):
                                    new_k = k.replace('fc1', 'fc1.experts.'+str(expert_idx))
                                    state_dict[new_k] = v
                            else:
                                state_dict[k] = v
                        elif 'fc2' in k:
                            if blocks_ind in self.MoE_Block_inds_fc2[stage_ind]:
                                for expert_idx in range(self.num_experts):
                                    new_k = k.replace('fc2', 'fc2.experts.'+str(expert_idx))
                                    state_dict[new_k] = v
                            else:
                                state_dict[k] = v
                    else:
                        state_dict[k] = v
                elif k.startswith('head'):
                    continue    
                else:
                    state_dict[k] = v
            
            # load state_dict 
            warnings = self.load_state_dict(state_dict, False)
            print(warnings)
            
            
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        gate_losses = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x,gate_loss = blk(x)
                if gate_loss is not None:
                    gate_losses.append(gate_loss)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        if len(gate_losses) > 0:
            return tuple(outs), sum(gate_losses)/len(gate_losses)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

@ROTATED_BACKBONES.register_module()
class VAN_moe_MultiInput(VAN_moe):
    def __init__(self,
                 in_channels=3,
                 datasets=None,
                 inject_uni_info_mode = None,
                 norm_cfg=None,
                 drop_path_rate=0.,
                 MoE_Block_inds_fc1=[[],[],[],[]],
                 MoE_Block_inds_fc2=[[],[],[],[]],
                 noisy_gating= True, 
                 num_experts= 2, 
                 gate= 'cosine', 
                 top_k= 2,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]
                ,img_size=256,
                embed_dims=[32, 64, 160, 256],
                mlp_ratios=[8, 8, 4, 4], drop_rate=0.,norm_layer=partial(nn.LayerNorm, eps=1e-6),
                depths=[3, 3, 5, 2], num_stages=4, 
                pretrained=None):
        super().__init__(
                MoE_Block_inds_fc1 = MoE_Block_inds_fc1,
                MoE_Block_inds_fc2 = MoE_Block_inds_fc2,
                num_experts=num_experts,
                top_k=top_k,
                img_size=img_size,
                noisy_gating= noisy_gating,
                gate= gate, 
                in_chans=in_channels,
                embed_dims=embed_dims,
                mlp_ratios=mlp_ratios,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                depths=depths,
                num_stages=num_stages, 
                pretrained=pretrained,
                init_cfg=init_cfg,
                norm_cfg=norm_cfg
                )
        self.init_datasets = datasets
       
        self.datasets = datasets if datasets is not None else ['single']
        
        self.inject_uni_info_mode = inject_uni_info_mode
        self.use_uni_head = inject_uni_info_mode is not None 
        if self.use_uni_head:
            assert self.datasets is not None
            self.datasets.append('uni_stem')
        if inject_uni_info_mode == 'scalar':
            self.alphas = [nn.Parameter(torch.tensor(0.5))] * len(self.init_datasets)
        if inject_uni_info_mode == 'channel_attention':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.se = nn.Sequential(
                nn.Linear(self.embed_dims[0], self.embed_dims[0] // 8, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims[0] // 8, self.embed_dims[0], bias=False),
                nn.Sigmoid()
            )

        self.dataset_stems = nn.ModuleDict()
        for dataset in self.datasets: 
            self.dataset_stems[dataset]=self.patch_embed1.proj
        if norm_cfg:
            self.patch_embed1 = build_norm_layer(norm_cfg, embed_dims[0])[1]
        else:
            self.patch_embed1 = nn.BatchNorm2d(embed_dims[0])
    def inject_uni_info(self, uni_feat, batch_input, datasets, split_sizes): 
        if self.inject_uni_info_mode == 'scalar': 
            uni_feat_split = torch.split(uni_feat, split_sizes, dim=0)
            for i,d in enumerate(datasets):
                batch_input[i] = self.scalar_inject_func(uni_feat_split[i], batch_input[i], d)  
            return batch_input
        if self.inject_uni_info_mode == 'channel_attention': 
            b, c, _, _ = uni_feat.size()
            y = self.avg_pool(uni_feat).view(b, c)
            y = self.se(y).view(b, c, 1, 1)
            ca = torch.split(y, split_sizes, dim=0)
            for i,d in enumerate(datasets):
                batch_input[i] = self.ca_inject_func(uni_feat[i], ca[i], batch_input[i])  
            return batch_input
    
    def scalar_inject_func(self, uni_feat, batch_input, dataset):  
        return  batch_input * self.alphas[self.init_datasets.index(dataset)] + uni_feat * (1-self.alphas[self.init_datasets.index(dataset)])

    def ca_inject_func(self, uni_feat, ca, batch_input ):  
        res = batch_input*ca.expand_as(batch_input) + uni_feat*(1-ca).expand_as(batch_input)
        return res
    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        gate_losses = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            if i==0:
                x=patch_embed(x)
                _,_,H,W=x.shape
            else: 
                x, H, W = patch_embed(x)
            for blk in block:
                x,gate_loss = blk(x)
                if gate_loss is not None:
                    gate_losses.append(gate_loss)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        if len(gate_losses) > 0:
            return tuple(outs), sum(gate_losses)/len(gate_losses)
        return tuple(outs)
    def forward(self, x, datasets=['single']):
        outs = []
        # assert len(x) == self.datasets
        batch_input = []
        if len(datasets) == 1:
            x = [x]
        
        if self.init_datasets is not None:
            for dataset, dataset_batch in (zip(datasets, x)):
                batch_input.append(self.dataset_stems[dataset](dataset_batch))
        else: 
            # assert False, (len(x), x[0].shape,x[1].shape)
            # print(len(x), x[0].shape,x[1].shape)
            x = torch.cat(x,dim=0)
            batch_input = [self.dataset_stems['single'](x)]

        if self.use_uni_head:
            split_sizes = [x_.size(0) for x_ in x]
            uni_x = torch.cat(x,dim=0) 
            uni_feat = self.dataset_stems['uni_stem'](uni_x)

            batch_input = self.inject_uni_info(uni_feat, batch_input, datasets,split_sizes)
        
        x = torch.cat(batch_input, dim=0)
        return self.forward_features(x)
 


    def init_weights(self):
        logger = MMLogger.get_current_instance()
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('block'):
                    if 'fc' in k:
                        k = k[5:]
                        stage_splits = k.split('.') 
                        stage_ind = eval(stage_splits[0])-1
                        blocks_ind = eval(stage_splits[1]) 
                        # if blocks_ind in [list(range(self.depth))[q] for q in self.MoE_Block_inds[stage_ind] if q < self.depth]:
                        k='block'+k
                        if 'fc1' in k:
                            if blocks_ind in self.MoE_Block_inds_fc1[stage_ind]:
                                for expert_idx in range(self.num_experts):
                                    new_k = k.replace('fc1', 'fc1.experts.'+str(expert_idx))
                                    state_dict[new_k] = v
                            else:
                                state_dict[k] = v
                        elif 'fc2' in k:
                            if blocks_ind in self.MoE_Block_inds_fc2[stage_ind]:
                                for expert_idx in range(self.num_experts):
                                    new_k = k.replace('fc2', 'fc2.experts.'+str(expert_idx))
                                    state_dict[new_k] = v
                            else:
                                state_dict[k] = v
                    else:
                        state_dict[k] = v
                elif k.startswith('patch_embed1'):
                    if 'norm' in k:
                        new_k=k.replace('.norm.','.')
                        state_dict[new_k]=v
                    else:
                        for i in self.datasets:
                            new_k = k.replace('patch_embed1.proj', 'dataset_stems.'+str(i))
                            state_dict[new_k] = v 
                elif k.startswith('head'):
                    continue    
                else:
                    state_dict[k] = v
            
            # load state_dict 
            warnings = self.load_state_dict(state_dict, False)
            print(warnings)

            if self.inject_uni_info_mode == 'channel_attention': 
                nn.init.constant_(self.se[2].weight, 0.0)
                if self.se[2].bias is not None:
                    nn.init.constant_(self.se[2].bias, 0.0)