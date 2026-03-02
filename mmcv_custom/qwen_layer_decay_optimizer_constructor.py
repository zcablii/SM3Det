import json
import torch.nn as nn
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.runner import get_dist_info

def get_num_layer_for_qwen(var_name, num_max_layer):
    """
    Assign layer ID for each parameter in the Qwen2.5 Vision Transformer.
    """
    # 属于 embedding 和 rotary embedding 的参数，层级为 0
    if var_name.startswith("backbone.patch_embed") or var_name.startswith("backbone.rotary_pos_emb"):
        return 0
    # 属于 transformer blocks 的参数
    elif var_name.startswith("backbone.blocks"):
        # e.g., var_name: backbone.blocks.15.mlp.fc2.bias
        # 通过 split 获取 block 的索引
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    # 其他参数（例如 neck, head, 或模型中其他未被匹配的参数）
    # 将被视为最深的层，使用基础学习率，不进行衰减
    else:
        return num_max_layer - 1


@OPTIMIZER_BUILDERS.register_module()
class QwenLayerDecayOptimizerConstructor(DefaultOptimizerConstructor):
    """
    Optimizer constructor for Qwen2.5 Vision Transformer with layer-wise
    learning rate decay.
    """
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module.
        """
        parameter_groups = {}
        # 从配置中获取层数和衰减率
        num_layers = self.paramwise_cfg.get('num_layers')
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        # BEiT/MAE style +2, see:
        # https://github.com/microsoft/unilm/blob/master/beit/run_class_finetuning.py#L272
        num_max_layer = num_layers + 2
        
        rank, _ = get_dist_info()
        if rank == 0:
            print(f"Build QwenLayerDecayOptimizerConstructor: "
                  f"layer_decay_rate={layer_decay_rate}, num_layers={num_layers}")

        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # 跳过冻结的权重

            # 为不需要权重衰减的参数分组 (e.g., bias, 1D params)
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            
            # 获取当前参数的层级 ID
            layer_id = get_num_layer_for_qwen(name, num_max_layer)
            
            # 创建唯一的组名，例如 "layer_5_decay"
            group_name = f"layer_{layer_id}_{group_name}"

            # 如果是新的组，则创建它
            if group_name not in parameter_groups:
                # 计算学习率的缩放比例
                scale = layer_decay_rate**(num_max_layer - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * self.base_lr,
                }
            
            # 将参数和其名称添加到对应的组中
            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)

        # 打印参数分组信息（仅在主进程）
        # if rank == 0:
        #     to_display = {}
        #     for key in sorted(parameter_groups.keys()):
        #         to_display[key] = {
        #             "param_names": parameter_groups[key]["param_names"],
        #             "lr_scale": f"{parameter_groups[key]['lr_scale']:.3f}",
        #             "lr": f"{parameter_groups[key]['lr']:.6f}",
        #             "weight_decay": parameter_groups[key]['weight_decay'],
        #         }
        #     print("Param groups = %s" % json.dumps(to_display, indent=2))

        params.extend(parameter_groups.values())