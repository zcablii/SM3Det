# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.distributions.normal import Normal

from mmcv.cnn import build_norm_layer, Linear, build_activation_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, _load_checkpoint
from mmcv.utils import to_2tuple
from mmengine.model import ModuleList, Sequential  
from mmengine.logging import MMLogger 
from mmengine.runner.checkpoint import CheckpointLoader

from mmdet.models.utils.transformer import PatchEmbed, PatchMerging
from ..builder import ROTATED_BACKBONES




class Conv3x3_FFN(nn.Module):
    def __init__(self, 
                 embed_dims=256,
                 feedforward_channels=1024,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,):
        super().__init__() 
        self.embed_dims = embed_dims
        self.k = 3
        self.feedforward_channels = feedforward_channels 
 
        conv1 = [nn.Conv2d(embed_dims, feedforward_channels, self.k),
                    build_activation_layer(act_cfg), 
                    nn.Dropout(ffn_drop),]
        self.conv1 = Sequential(*conv1)
        ffn = [Linear(feedforward_channels, embed_dims),nn.Dropout(ffn_drop)]
        self.ffn = Sequential(*ffn)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

        self.gamma2 = nn.Identity()

    def forward(self, x, identity=None):
        out = self.conv1(x).squeeze(-1).squeeze(-1) 
        out = self.ffn(out)
        out = self.gamma2(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x[:,:,(self.k-1)//2,(self.k-1)//2] 
        return identity + self.dropout_layer(out)

class Conv5x5_FFN(Conv3x3_FFN):
    def __init__(self, 
                 embed_dims=256,
                 feedforward_channels=1024,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,):
        super().__init__(embed_dims,
                 feedforward_channels,
                 act_cfg,
                 ffn_drop,
                 dropout_layer,
                 add_identity) 
        self.k = 5
        conv1 = [nn.Conv2d(embed_dims, feedforward_channels, self.k),
                    build_activation_layer(act_cfg), 
                    nn.Dropout(ffn_drop),]
        self.conv1 = Sequential(*conv1)


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

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, 
                 moe_cfg
                 ):
        super(MoE_layer, self).__init__() 

        self.noisy_gating = moe_cfg['noisy_gating']
        self.input_size = moe_cfg['embed_dims']
        self.k = moe_cfg['top_k']
        self.gating = moe_cfg['gating']
        # instantiate experts
        self.num_experts = moe_cfg['num_experts']
        self.squad_num = None
        self.squads = None

        if 'squad_num' in moe_cfg.keys():
            self.squad_num = moe_cfg['squad_num']
            self.squads = moe_cfg['squads']
            self.num_experts = self.squad_num * len(self.squads)

            self.experts  = nn.ModuleList([])
            for squad_idx in range(self.squad_num):
                for each in self.squads:
                    net = eval(each)
                    self.experts .append(net(
                        embed_dims=self.input_size,
                        feedforward_channels=moe_cfg['feedforward_channels'], 
                        ffn_drop=moe_cfg['drop_rate'],
                        dropout_layer=dict(type='DropPath', drop_prob=moe_cfg['drop_path_rate']),
                        act_cfg=moe_cfg['act_cfg'],
                        add_identity=True,
                    ))            
            
        else:
            self.experts = nn.ModuleList([FFN(
                embed_dims=self.input_size,
                feedforward_channels=moe_cfg['feedforward_channels'],
                num_fcs=2,
                ffn_drop=moe_cfg['drop_rate'],
                dropout_layer=dict(type='DropPath', drop_prob=moe_cfg['drop_path_rate']),
                act_cfg=moe_cfg['act_cfg'],
                add_identity=True,
                init_cfg=None) for i in range(self.num_experts)]) 
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
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        if len(x.shape) == 2:
            x = x.sum(dim=0)
        return x.float().var() / (x.float().mean()**2 + eps)


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

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
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        #clean_logits = self.w_gate(x)
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


    def forward(self, x, identity=None, loss_coef=1e-2, hwshape=None):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        train = self.training
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0) 
            identity = identity.unsqueeze(0)
        x_shape = x.shape
        x = x.reshape(-1,x.shape[-1])
        identity = identity.reshape(-1,identity.shape[-1]) 
        gates, load = self.noisy_top_k_gating(x, train) 
        # calculate importance loss
        importance = gates.sum(dim=0)
        
        # calculate loss
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates, self.squads ) 
        expert_inputs,identity = dispatcher.dispatch(x,identity, shape=[x_shape[0], *hwshape, x_shape[-1]]) 
        gates = dispatcher.expert_to_gates() 
        expert_outputs = [self.experts[i](expert_inputs[i], identity[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs).reshape(x_shape)

        return y, loss
     
class SparseDispatcher(object):
    def __init__(self, num_experts, gates, squads=None):
        """Create a SparseDispatcher."""
        
        # With pytorch, I have an image feature tensor and a convolution layer. 
        # I want to perform sparse convolution to improve efficiency. 
        # How to implement it if I give a list of index on the feature map where need to perform convolution and others not?
        self.squads = None
        if squads:
            assert num_experts%len(squads)==0
            self.squads = squads*(num_experts//len(squads))
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

    def dispatch(self, inp, identity=None, shape=None):
        
       
        if identity is not None:
            identity = identity[self._batch_index].squeeze(1)
        inp_exp = inp[self._batch_index].squeeze(1)

        if self.squads is None:
            return torch.split(inp_exp, self._part_sizes, dim=0), torch.split(identity, self._part_sizes, dim=0)
        
        else:
            # def get_kernel_feat(ind, k=3):
            #     b,h,w,d = shape
            #     i, j = ind %(h*w)// w, ind % w
            #     x_unflattened = inp.view(b, h, w, d)
            #     pad_margin = tuple((k-1)//2 for i in range(4))
            #     padded_x = torch.nn.functional.pad(x_unflattened, (0, 0, *pad_margin))  # pad with 1 zero on each side
            #     neighbor_features = padded_x[ind//(h*w), i: i+k, j:j+k, :] 
            #     return neighbor_features
            

            split_idx = torch.split(self._batch_index, self._part_sizes, dim=0)
            splits = []
            for i, sq in enumerate(self.squads):
                assert sq in ['FFN', 'Conv3x3_FFN',  'Conv5x5_FFN']
                if sq == 'FFN':
                    splits.append(inp[split_idx[i]])
                else:
                    # ys = []
                    # for i in split_idx[i]: 
                    #     y = get_kernel_feat(i,3 if sq == 'Conv3x3_FFN' else 5)
                    #     ys.append(y.unsqueeze(0))
                    # ys = torch.cat(ys,dim=0).permute(0,3,1,2).contiguous()
                    # splits.append(ys)
                    k = 3 if sq == 'Conv3x3_FFN' else 5
                    inp_ = torch.cat([inp, torch.zeros(1,shape[-1], device=inp.device)], dim=0)
                    b,h,w,d = shape
                    x_idx = torch.tensor(range(b*h*w), device=inp.device).reshape(b, h, w).unsqueeze(-1)

                    def get_kernel_feat(ind, k=3):
                
                        i, j = ind %(h*w)// w, ind % w
                        pad_margin = tuple((k-1)//2 for i in range(4)) 
                        padded_x = torch.nn.functional.pad(x_idx, (0, 0, *pad_margin), "constant", -1).squeeze(-1)  # pad with 1 zero on each side
                        neighbor_features = padded_x[ind//(h*w), i: i+k, j:j+k] 
                        return neighbor_features 
            
                    result = torch.stack([get_kernel_feat(x,k) for x in split_idx[i]], dim=0)
                    result_ = result.flatten() 
                    ys = inp_[result_].reshape(len(result),k,k,shape[-1]).permute(0,3,1,2).contiguous()
                    splits.append(ys)

            return splits, torch.split(identity, self._part_sizes, dim=0)
                


    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        
        #stitched = torch.cat(expert_out, 0).exp()
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
        # add eps to all zero values in order to avoid nans when going back to log space
        
        #combined[combined == 0] = np.finfo(float).eps
        # back to log space
        #return combined.log()
        return combined


    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)
class ShiftWindowMSA(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

class SwinBlock(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 MoE_cfg=None,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__()
        self.init_cfg = init_cfg
        self.with_cp = with_cp
        self.MoE_cfg = MoE_cfg
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        
        if MoE_cfg is not None:
            MoE_cfg.update({'embed_dims': embed_dims, 
                'feedforward_channels': feedforward_channels, 
                'drop_rate': drop_rate,'drop_path_rate': drop_path_rate, 
                'act_cfg':act_cfg})
            self.ffn = MoE_layer(MoE_cfg)
        else:
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=2,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                add_identity=True,
                init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            loss = None
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x 
            x = self.norm2(x) 
            if self.MoE_cfg:
                x = self.ffn(x, identity=identity, hwshape=hw_shape)
            else:
                x = self.ffn(x, identity=identity)

            if type(x) == tuple:
                x, loss = x

            return x, loss

        if self.with_cp and x.requires_grad:
            x, loss = cp.checkpoint(_inner_forward, x)
        else:
            x, loss = _inner_forward(x)

        return x, loss
    def init_weights(self):
        pass


class SwinBlockSequence(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 MoE_Block_ind = [],
                 noisy_gating= True, 
                 num_experts= 4, 
                 top_k= 2,
                 gate = 'cosine',
                 squads = None,
                 squad_num = 0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        self.MoE_Block_ind = [list(range(depth))[i] for i in MoE_Block_ind if i < depth]
        self.blocks = ModuleList()
        for i in range(depth):
            MoE_cfg = None
            if i in self.MoE_Block_ind:
                MoE_cfg = {'noisy_gating': noisy_gating, 'num_experts': num_experts, 'top_k': top_k, 'gating': gate}
                if squads is not None and squad_num>0:
                    MoE_cfg.update({'squads': squads, 'squad_num':squad_num})
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                MoE_cfg = MoE_cfg,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        losses = []
        for block in self.blocks:
            x, loss = block(x, hw_shape)
            if loss is not None:
                losses.append(loss) 

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape, losses
        else:
            return x, hw_shape, x, hw_shape, losses


@ROTATED_BACKBONES.register_module()
class SwinTransformer_MoE(BaseModule):
    
    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 MoE_Block_inds = [[],[],[],[]],
                 noisy_gating= True, 
                 num_experts= 4, 
                 gate= 'cosine', 
                 top_k= 2,
                 squads = None, # ['FFN', 'Conv3x3_FFN',  'Conv5x5_FFN']
                 squad_num = 0,
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SwinTransformer_MoE, self).__init__(init_cfg=init_cfg)

        
        self.MoE_Block_inds = MoE_Block_inds
        self.num_experts= num_experts
        self.squads = squads
        self.squad_num = squad_num
        if squads is not None and squad_num>0:
            self.num_experts= squad_num*len(squads)
        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                MoE_Block_ind = MoE_Block_inds[i],
                noisy_gating= noisy_gating, 
                gate = gate,
                num_experts= num_experts, 
                top_k= top_k,
                squads = squads,
                squad_num = squad_num,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_MoE, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


    def forward(self, x, datasets=['single']):
        # assert len(x) == self.datasets
        
        if len(datasets) == 1:
            x = [x]
        x = torch.cat(x,dim=0)

        x, hw_shape = self.patch_embed(x) # hw_shape: (out_h, out_w)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        gate_losses = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape, loss = stage(x, hw_shape)
            if len(loss)>0:
                gate_losses = gate_losses + loss
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs, sum(gate_losses)/len(gate_losses)

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
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = self.swin_converter(_state_dict) 
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            miss,unexp = self.load_state_dict(state_dict, False)
            # print(miss)
            # print('####################=============')
            # print(unexp)



    def swin_converter(self, ckpt):
        import re
        new_ckpt = OrderedDict()
        def init_conv_from_ffn(ffn, k):
            conv = nn.Conv2d(*ffn.weight.shape,k) 
            w_ = torch.zeros(conv.weight.shape)
            center = (k-1)//2
            w_[:,:,center,center] = ffn.weight
            conv.weight = nn.Parameter(w_) 
            # print(conv.bias)
            conv.bias =ffn.bias
            return conv
        def correct_unfold_reduction_order(x):
            out_channel, in_channel = x.shape
            x = x.reshape(out_channel, 4, in_channel // 4)
            x = x[:, [0, 2, 1, 3], :].transpose(1,
                                                2).reshape(out_channel, in_channel)
            return x

        def correct_unfold_norm_order(x):
            in_channel = x.shape[0]
            x = x.reshape(4, in_channel // 4)
            x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
            return x
 

        for k, v in ckpt.items():
            if k.startswith('head'):
                continue
            elif k.startswith('layers'):
                new_v = v
                if 'attn.' in k:
                    new_k = k.replace('attn.', 'attn.w_msa.')
                elif 'mlp.' in k:
                    stage_splits = re.split(r'layers.|.blocks.|.mlp.fc*', k)
                    if len(stage_splits)==4: # ['', '2', '11', '2.weight']
                        stage_ind = eval(stage_splits[1])
                        blocks_ind = eval(stage_splits[2]) 
                        if not blocks_ind in self.stages[stage_ind].MoE_Block_ind:
                           
                            if 'mlp.fc1.' in k:
                                new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                            elif 'mlp.fc2.' in k:
                                new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                            else:
                                new_k = k.replace('mlp.', 'ffn.')
                        else:
                            for expert_idx in range(self.num_experts):
                                # self.squads 
                                # self.squad_num  
                                # w_ = torch.zeros(conv_ffn.conv1[0].weight.shape)
                                # print('shape',w_.shape) # [out_dim, in_dim, k, k]
                                # center = (k-1)//2
                                # w_[:,:,center,center] = ffn.layers[0][0].weight
                                # conv_ffn.conv1[0].weight = nn.Parameter(w_) 
                                # conv_ffn.conv1[0].bias = ffn.layers[0][0].bias
                                
                                # conv_ffn.ffn[0].weight = ffn.layers[1].weight
                                # conv_ffn.ffn[0].bias =ffn.layers[1].bias
                                new_v_ = new_v
                                if self.squads is not None and self.squad_num > 0:
                                    squads = self.squads * self.squad_num
                                    if 'mlp.fc1.' in k:
                                        if squads[expert_idx] == 'FFN':
                                            sub_net = '.layers.0.0.'
                                        elif squads[expert_idx] in ['Conv3x3_FFN','Conv5x5_FFN']:
                                            sub_net = '.conv1.0.'
                                            if 'bias' not in k:
                                                kn = 3 if '3' in squads[expert_idx] else 5
                                                w_ = torch.zeros(*new_v_.shape,kn,kn) 
                                                center = (kn-1)//2 
                                                w_[:,:,center,center] = new_v_
                                                new_v_ = nn.Parameter(w_) 
                                        else:
                                            assert False, squads[expert_idx]
                                        new_k = k.replace('mlp.fc1.', 'ffn.experts.'+str(expert_idx)+sub_net)
                                    elif 'mlp.fc2.' in k: 
                                        if squads[expert_idx] == 'FFN':
                                            sub_net = '.layers.1.'
                                        elif squads[expert_idx] in ['Conv3x3_FFN','Conv5x5_FFN']:
                                            sub_net = '.ffn.0.'
                                        else:
                                            assert False, squads[expert_idx]
                                        new_k = k.replace('mlp.fc2.', 'ffn.experts.'+str(expert_idx)+sub_net)

                                else:
                                    if 'mlp.fc1.' in k:
                                        new_k = k.replace('mlp.fc1.', 'ffn.experts.'+str(expert_idx)+'.layers.0.0.')
                                    elif 'mlp.fc2.' in k: 
                                        new_k = k.replace('mlp.fc2.', 'ffn.experts.'+str(expert_idx)+'.layers.1.')
                                new_k = new_k.replace('layers', 'stages', 1)
                                new_ckpt['backbone.' + new_k] = new_v_
                            continue
                             
                elif 'downsample' in k:
                    new_k = k
                    if 'reduction.' in k:
                        new_v = correct_unfold_reduction_order(v)
                    elif 'norm.' in k:
                        new_v = correct_unfold_norm_order(v)
                else:
                    new_k = k
                new_k = new_k.replace('layers', 'stages', 1)
            elif k.startswith('patch_embed'):
                new_v = v
                if 'proj' in k:
                    new_k = k.replace('proj', 'projection')
                else:
                    new_k = k
            else:
                new_v = v
                new_k = k

            new_ckpt['backbone.' + new_k] = new_v

        return new_ckpt

