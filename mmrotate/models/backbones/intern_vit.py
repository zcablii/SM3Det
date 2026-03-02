# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from safetensors import safe_open
try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    # assert False, 'FlashAttention is not installed.'
    has_flash_attn = False

# logger = logging.get_logger(__name__)
class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


NORM2FN = {
    'rms_norm': InternRMSNorm,
    'layer_norm': nn.LayerNorm,
}


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



try:
    from apex.normalization import FusedRMSNorm

    RMSNorm = FusedRMSNorm  # noqa

    print('Discovered apex.normalization.FusedRMSNorm - will use it instead of RMSNorm')
except ImportError:
    # using the normal RMSNorm
    pass
except Exception:
    print('discovered apex but it failed to load, falling back to RMSNorm')
    pass


class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size

        self.causal = causal
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()

    def forward(self, x, H, W):
        # cls, x = x[:, :1, :], x[:, 1:, :]
        B, N, C = x.shape
        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size
        x = x.view(B, H, W, C)
        x = F.pad(x, [0, 0, 0, W_ - W, 0, H_ - H])
        
        x = window_partition(x, window_size=self.window_size)  # nW*B, window_size, window_size, C
        x = x.view(-1, N_, C)
        
        def _naive_attn(x):
            qkv = self.qkv(x).view(-1, N_, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            if self.qk_normalization:
                B_, H_, _, D_ = q.shape
                q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
                k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, num_head, N_, N_]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
            x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size, self.window_size, C)
            return x
        
        def _flash_attn(x):
            qkv = self.qkv(x).view(-1, N_, 3, self.num_heads, C // self.num_heads)
            
            if self.qk_normalization:
                q, k, v = qkv.unbind(2)
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
                qkv = torch.stack([q, k, v], dim=2)
                
            # qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
            context, _ = self.inner_attn(qkv, causal=self.causal)
            x = context.reshape(-1, self.window_size, self.window_size, C)
            # x = rearrange(context, "b s h d -> b s (h d)").reshape(-1, self.window_size, self.window_size, C)
            return x
        
        x = _naive_attn(x) if not self.use_flash_attn else _flash_attn(x)
        x = x.contiguous()
        
        x = window_reverse(x, self.window_size, H_, W_)
        x = x[:, :H, :W, :].reshape(B, N, C).contiguous()
        # x = torch.cat([cls, x], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class InternAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False,layer_norm_eps=1e-06):
        super().__init__()
        # print(use_flash_attn)
       
        self.embed_dim=dim
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn and has_flash_attn
        head_dim = dim // num_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_normalization = qk_normalization
        
        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=layer_norm_eps)
        # print(use_flash_attn)
        if use_flash_attn==True:
            # self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)
        self.proj = nn.Linear(dim, dim)
    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
        )
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, x, H=None, W=None):
        # x = self._flash_attn(x)
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
        return x


class InternMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, hidden_size, intermediate_size, act_layer=nn.GELU):
        super().__init__()
        self.act = act_layer()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class InternVisionEncoderLayer(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, with_cp=False,
            qk_normalization=False, layerscale_force_fp32=False, windowed=False, window_size=14,layer_norm_eps=1e-06):
        super().__init__()

        
        if windowed:
            self.attn = WindowedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                          proj_drop=drop, use_flash_attn=use_flash_attn, causal=False,
                                          norm_layer=norm_layer, qk_normalization=qk_normalization,
                                          window_size=window_size)
        else:
            self.attn = InternAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                  proj_drop=drop, use_flash_attn=use_flash_attn, causal=False,
                                  norm_layer=norm_layer, qk_normalization=qk_normalization)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        

        self.embed_dim=dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = InternMlp(hidden_size=dim, intermediate_size=mlp_hidden_dim, act_layer=act_layer)
        self.norm1 = norm_layer(dim,layer_norm_eps)
        self.norm2 = norm_layer(dim,layer_norm_eps)
        self.ls1 = nn.Parameter(init_values * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.with_cp = with_cp

    def forward(self, x, H=None, W=None):

        def _inner_forward(x, H, W):
            x = x + self.drop_path1(self.attn(self.norm1(x), H, W)*self.ls1)
            x = x + self.drop_path2(self.mlp(self.norm2(x))*self.ls2)
            return x
        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, H, W)
        else:
            return _inner_forward(x, H, W)
        # return _inner_forward(x, H, W)


class InternVisionEmbeddings(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, flatten=True):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.img_size // self.patch_size, self.img_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed
    def forward_adapter(self, x):
        patch_embeds = self.patch_embedding(x)
        _, _, H, W = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        bs, n, dim = patch_embeds.shape
        x = patch_embeds + self.position_embedding[:, 1:]#.to(target_dtype)
        return x, H, W, bs, n, dim

    def forward(self, x):
        target_dtype = self.patch_embedding.weight.dtype
        # print(target_dtype)
        patch_embeds = self.patch_embedding(x)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


@BACKBONES.register_module()
class InternViT(BaseModule):

    def __init__(self, in_chans=3, patch_size=14, img_size=896, pretrain_size=448, qkv_bias=True, drop_path_rate=0.1,
                 embed_dim=1024, num_heads=16, mlp_ratio=4, init_values=0.1, qk_normalization=True, depth=24,
                 use_flash_attn=True, with_cp=True, layerscale_force_fp32=False, out_indices=[7, 11, 15, 23],
                 freeze_vit=False, with_fpn=False, with_simple_fpn=False,last_feat=False, with_final_norm=False, window_attn=False, window_size=14,
                 output_dtype="float32",norm_type='layer_norm',pretrained=None,pretrained_type='ViT', lora_config=None):

        super().__init__()

        use_flash_attn = [use_flash_attn] * depth if not isinstance(use_flash_attn, list) else use_flash_attn
        window_attn = [window_attn] * depth if not isinstance(window_attn, list) else window_attn
        window_size = [window_size] * depth if not isinstance(window_size, list) else window_size
        self.embeddings = InternVisionEmbeddings(img_size, patch_size, in_chans, embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # self.encoder = 
        # stochastic depth decay rule
        self.last_feat=last_feat
        self.norm_layer = NORM2FN[norm_type]
        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=NORM2FN[norm_type],
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,drop=0.,
                  use_flash_attn=use_flash_attn[i],
                  with_cp=with_cp,
                  qk_normalization=qk_normalization,
                  layerscale_force_fp32=layerscale_force_fp32,
                  windowed=window_attn[i],
                  window_size=window_size[i],layer_norm_eps=1e-6) for i in range(depth)])
    
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pretrain_size = pretrain_size
        self.drop_path_rate = drop_path_rate
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_indices = out_indices
        self.with_fpn = with_fpn
        self.with_simple_fpn = with_simple_fpn
        if output_dtype == 'float16':
            self.output_dtype = torch.float16
        elif output_dtype == 'bfloat16':
            self.output_dtype = torch.bfloat16
        elif output_dtype == 'float32':
            self.output_dtype = torch.float32
        else:
            raise NotImplementedError

        self.init_weights(pretrained,pretrained_type)
        
        self.freeze_vit = freeze_vit
        
        if self.freeze_vit:
            self._freeze_params()
        if with_fpn:
            self.fpn1 = nn.Sequential(*[
                nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
                NORM2FN[norm_type](embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
                NORM2FN[norm_type](embed_dim) if with_final_norm else nn.Identity()
            ])
            self.fpn2 = nn.Sequential(*[
                nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
                NORM2FN[norm_type](embed_dim) if with_final_norm else nn.Identity()
            ])
            self.fpn3 = nn.Sequential(*[
                nn.Identity(),
                NORM2FN[norm_type](embed_dim) if with_final_norm else nn.Identity()
            ])
            self.fpn4 = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=2, stride=2),
                NORM2FN[norm_type](embed_dim) if with_final_norm else nn.Identity()
            ])
            self.fpn1.apply(self._init_weights)
            self.fpn2.apply(self._init_weights)
            self.fpn3.apply(self._init_weights)
            self.fpn4.apply(self._init_weights)
        elif with_simple_fpn:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            logger = get_root_logger()
            logger.info('Build model without FPN.')
            # self.up1.apply(self._init_weights)
            # self.up2.apply(self._init_weights)
            # self.up3.apply(self._init_weights)
            # self.up4.apply(self._init_weights)
        self.gradient_checkpointing = True

        
        if lora_config:
            from peft import LoraConfig, get_peft_model
            target_modules = []
            for name, module in self.named_modules():
                if any(n in name for n in ["fc1","fc2"]):
                    target_modules.append(name)
            # 2. 生成 LoRA 配置
            lora_cfg = LoraConfig(
                r=lora_config['r'],
                lora_alpha=lora_config['lora_alpha'],
                use_dora=lora_config.get('use_dora', False),
                use_rslora=lora_config.get('use_rslora', False),
                lora_dropout=lora_config.get('lora_dropout', 0.05),
                bias=lora_config.get('bias', 'none'),
                task_type=lora_config.get('task_type', None),
                target_modules=target_modules,
            )
            # 3. 原地替换
            self = get_peft_model(self, lora_cfg)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def _freeze_params(self):
        logger = get_root_logger()
        for name, param in self.named_parameters():
            if 'embeddings' in name or 'layers' in name:
                param.requires_grad = False
                logger.info(f'{name} is freezed!')
            else:
                logger.info(f'{name} is not freezed!')

        # Check if embeddings and layers exist
        if hasattr(self, 'embeddings'):
            self.embeddings.eval()
            logger.info('embeddings module is set to eval mode!')
        else:
            logger.warning('embeddings module does not exist!')

        if hasattr(self, 'layers'):
            self.layers.eval()
            logger.info('layers module is set to eval mode!')
        else:
            logger.warning('layers module does not exist!')


    def init_weights(self, pretrained=None,pretrained_type='ViT'):

        def resize_pos_embed(pos_embed, H, W):
            cls = pos_embed[:, :1, :]
            pos_embed = pos_embed[:, 1:, :].reshape(
                1, self.pretrain_size // 14, self.pretrain_size // 14, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed.float(), size=(H, W), mode='bicubic', align_corners=False).to(cls.dtype). \
                reshape(1, -1, H * W).permute(0, 2, 1)
            pos_embed = torch.cat([cls, pos_embed], dim=1)
            pos_embed = nn.Parameter(pos_embed)
            return pos_embed

        if isinstance(pretrained, str):
            logger = get_root_logger()
            if pretrained_type=='ViT':
                
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'module' in checkpoint:
                    checkpoint = checkpoint['module']
            else:
                with safe_open(pretrained, framework="pt") as f:
                    state_dict = {key: f.get_tensor(key) for key in f.keys()}
                    checkpoint = {}
                    for key in state_dict:
                        if key.startswith("vision_model") or key.startswith("encoder") or key.startswith("embeddings"):
                            new_key = key.replace("vision_model.", "").replace("encoder.", "")#.replace("embeddings.","")
                            new_key = new_key.replace("embedding.proj.","embedding.")
                            #new_key = new_key.replace("embedding","embed").replace("patch_embed","patch_embed.proj").replace("position","pos")
                            checkpoint[new_key] = state_dict[key]
                # resize pos_embed
            # print({key for key in checkpoint.keys()})
            pos_embed = checkpoint['embeddings.position_embedding']
            checkpoint['embeddings.position_embedding'] = resize_pos_embed(
                pos_embed, self.img_size // self.patch_size, self.img_size // self.patch_size)
            # resize patch_embed
            patch_embed = checkpoint['embeddings.patch_embedding.weight']
            checkpoint['embeddings.patch_embedding.weight'] = F.interpolate(
                patch_embed, size=(self.patch_size, self.patch_size),
                mode='bicubic', align_corners=False)
            message = self.load_state_dict(checkpoint, strict=False)
            logger.info(message)
    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.img_size // self.patch_size, self.img_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed.float(), size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1).to(pos_embed.dtype)
        return pos_embed

    @property
    def dtype(self):
        return self.embeddings.patch_embedding.weight.dtype
    
    def forward_encoder(self, x):
        for blk in self.layers:
            x = blk(x)
        return x
    
    def forward(self, x, datasets=['single']):
        if len(datasets) == 1:
            x = [x]
        x = torch.cat(x,dim=0)
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x = self.embeddings(x)
        # print(x)
        features = []
        for i, blk in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)       
                features.append(xp.contiguous())
            # print(x)
        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(ops)):
                features[i] = ops[i](features[i])
        if self.last_feat:
            return tuple(features), x
        return tuple(features)
    

    def train(self, mode=True):
        """Train function of  ReResNet."""
        super(InternViT, self).train(mode)
        if self.freeze_vit:
            self._freeze_params()