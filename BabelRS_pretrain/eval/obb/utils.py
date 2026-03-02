"""
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Credit: 
https://github.com/mlfoundations/open_clip/blob/main/src/training/distributed.py
https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/distributed.py
"""
import os
import re
import argparse
import subprocess
from contextlib import nullcontext

import torch
import torch.distributed as dist


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    # chunk_size = math.ceil(len(lst) / n)  # integer division
    # return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
    base_chunk_size = len(lst) // n  # integer division
    remainder = len(lst) % n  # remaining elements
    chunks = []
    for i in range(n):
        chunk_size = base_chunk_size + (i < remainder)  # add one to the chunk size for the first 'remainder' chunks
        start = i * base_chunk_size + min(i, remainder)  # calculate the start index
        chunks.append(lst[start:start+chunk_size])
    return chunks


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    distributed_type = None
    if world_size > 1:
        if "SLURM_PROCID" in os.environ:
            distributed_type = "slurm"
        elif "OMPI_COMM_WORLD_RANK" in os.environ:
            distributed_type = "openmpi"
        elif "PMI_RANK" in os.environ:
            distributed_type = "pmi"
        elif "LOCAL_RANK" in os.environ:
            distributed_type = "torch"
    elif "SLURM_PROCID" in os.environ:
        distributed_type = "slurm"
    return local_rank, global_rank, world_size, distributed_type


def get_slurm_master_addr():
    nodelist = os.environ.get('SLURM_JOB_NODELIST')
    if nodelist is not None:
        cmd = f'scontrol show hostnames "{nodelist}" | head -n 1'
        master_addr = os.popen(cmd).read().strip()
        return master_addr
    return None


def init_distributed_device(dist_backend="nccl", dist_url="env://", no_set_device_rank=False):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    distributed = False
    world_size = 1
    rank = 0  # global rank
    local_rank = 0
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if is_using_distributed():
        if "SLURM_PROCID" in os.environ:
            # DDP via SLURM
            local_rank, rank, world_size, _ = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = get_slurm_master_addr()
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=world_size,
                rank=rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            local_rank, _, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend, init_method=dist_url
            )
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        distributed = True
    else:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        # needed to run on single gpu
        torch.distributed.init_process_group(
            backend=dist_backend,
            init_method=dist_url,
            world_size=1,
            rank=0,
        )

    if torch.cuda.is_available():
        if distributed and not no_set_device_rank:
            device = "cuda:%d" % local_rank
        else:
            device = "cuda:0"
        try:
            torch.cuda.set_device(device)
        except Exception as e:
            print(f"Failed to set device to {device}: {e}")
            import ipdb; ipdb.set_trace()
            raise e
    else:
        device = "cpu"
    return torch.device(device)


def get_cmd_output(cmd):
    return subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
"""
Credit: 
https://github.com/mlfoundations/open_clip/blob/main/src/training/distributed.py
https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/distributed.py
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
"""

class MyArgumentParser(argparse.ArgumentParser):
    def add_argument(self, option_string, *args, **kwargs):
        if not hasattr(self, 'argument_names_map'):
            self.argument_names_map = {}
        # Store the original argument name
        if "-" in option_string.strip("-"):
            tgt_string = option_string.strip("-").replace("-", "_")
            self.argument_names_map[tgt_string] = option_string.strip("-")
        super().add_argument(option_string, *args, **kwargs)
    
    def get_args_command_line(self, args):
        command_line = []
        for arg in vars(args):
            v = getattr(args, arg)
            if v is not None:
                if arg in self.argument_names_map:
                    arg = self.argument_names_map[arg]
                command_line.append('--' + arg)
                command_line.append(str(v))
        return ' '.join(command_line)


def monkey_patch_of_collections_typehint_for_mmrotate1x():
    import collections
    from collections.abc import Mapping, Sequence, Iterable
    collections.Mapping = Mapping
    collections.Sequence = Sequence
    collections.Iterable = Iterable


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)
        
        
def get_num_parameters(module):
    """Modified from print_trainable_parameters of peft"""
    def _get_parameter_numel(param):
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            # if using DS Zero 3 and the weights are initialized empty
            num_params = param.ds_numel
        return num_params
    
    if isinstance(module, torch.Tensor):  # nn.Parameter()
        num_params = _get_parameter_numel(module)
        return num_params if module.requires_grad else 0, num_params
        
    trainable_params = 0
    all_param = 0
    for param in module.parameters():
        num_params = _get_parameter_numel(param)
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param


def print_trainable_parameters(model):    
    trainable_params, all_param = get_num_parameters(model)
    rank0_print(f"[WHOLE_MODEL] trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")
    
    model_cls_name = type(model).__name__
    
    if bool(re.match(r"^Llava.*ForCausalLM$", model_cls_name)): # LlavaLlamaForCausalLM
        named_part_modules = {
            "VISION TOWER": model.get_vision_tower(), 
            "MULTIMODAL PROJ": model.get_model().mm_projector, 
            "LANGUAGE LAYERS": model.get_model().layers,
            "INPUT EMBEDDINGS": model.get_input_embeddings(), 
            "OUTPUT EMBEDDINGS": model.get_output_embeddings(),
        }
        if hasattr(model, "vision_resampler"):
            named_part_modules["VISION RESAMPLER"] = model.get_model().vision_resampler
        
    elif model_cls_name == "Florence2ForConditionalGeneration":
        named_part_modules = {
            "VISION TOWER": model.get_vision_tower(), 
            "MULTIMODAL PROJ": model.get_mm_projection(), 
            "LANGUAGE MODEL": model.get_language_model(),
        }
    elif model_cls_name == "Qwen2VLForConditionalGeneration":
        named_part_modules = {
            "VISION AND MERGER": model.visual,
            "LANGUAGE MODEL": model.model,
        }
    elif model_cls_name == "InternVLChatModel":
        named_part_modules = {
            "VISION TOWER": model.vision_model, 
            "MULTIMODAL PROJ": model.mlp1, 
            "LANGUAGE MODEL": model.language_model,
        }
    else:
        named_part_modules = {}
        
    for part_name, part_module in named_part_modules.items():
        trainable_params, all_param = get_num_parameters(part_module)
        rank0_print(f"[{part_name}] trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")


def check_pretrained_load(model, model_name_or_path):
    archive_file = os.path.join(model_name_or_path, "pytorch_model.bin.index.json")
    if not os.path.isfile(archive_file):
        rank0_print(f"[PRETRAINED LOADING CHECK] archive_file {archive_file} does not exist, skip checking. (pretrained path may be a hub url, whose checking is not implemented for this codebase)")
        return
    
    import gc
    from transformers.modeling_utils import load_state_dict as load_state_dict_from_checkpoint_file
    from transformers.integrations import is_deepspeed_zero3_enabled
    from transformers.utils.hub import get_checkpoint_shard_files
    
    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
        model_name_or_path,
        archive_file,
        resume_download=False,
        local_files_only=False,
        user_agent={'file_type': 'model', 'framework': 'pytorch', 'from_auto_class': False},
        revision='main',
        subfolder='',
    )
    
    _deepspeed_imported = False
    if is_deepspeed_zero3_enabled():
        if not _deepspeed_imported:
            import deepspeed
            _deepspeed_imported = True
        with deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=0):
            model_state_dict = {k:v.cpu() for k, v in model.state_dict().items()}
    else:
        model_state_dict = {k:v.cpu() for k, v in model.state_dict().items()}
    
    loaded_state_dict = {}
    for shard_file in resolved_archive_file:
        loaded_state_dict.update(load_state_dict_from_checkpoint_file(shard_file))
    
    missing_keys = [key for key in model_state_dict if key not in loaded_state_dict]
    unexpected_keys = [key for key in loaded_state_dict if key not in model_state_dict]
    mismatched_keys = []

    for key in model_state_dict.keys():
        if key in loaded_state_dict:
            model_p = model_state_dict[key]
            loaded_p = loaded_state_dict[key]
            if model_p.dtype != loaded_p.dtype:
                loaded_p = loaded_p.to(model_p.dtype)
            if not torch.allclose(model_p, loaded_p):
                mismatched_keys.append(key)
                
    if not missing_keys and not unexpected_keys and not mismatched_keys:
        rank0_print("[PRETRAINED LOADING CHECK] All pretrained parameters have been successfully loaded into the model.")
    else:
        if missing_keys:
            rank0_print(f"[PRETRAINED LOADING CHECK] The following parameters are missing in the pretrained state dict and could not be loaded: {missing_keys}")
        if unexpected_keys:
            rank0_print(f"[PRETRAINED LOADING CHECK] The following pretrained parameters were not found in the model and are considered extra: {unexpected_keys}")
        if mismatched_keys:
            rank0_print(f"[PRETRAINED LOADING CHECK] The following parameters could not be correctly loaded due to not allclose: {mismatched_keys}")
    
    # force memory release
    del model_state_dict
    del loaded_state_dict
    gc.collect()


def maybe_zero3_gathered_parameters(tensor, modifier_rank=0):
    import deepspeed
    from transformers.integrations import is_deepspeed_zero3_enabled
    if is_deepspeed_zero3_enabled():
        return deepspeed.zero.GatheredParameters(tensor, modifier_rank=modifier_rank)
    else:
        return nullcontext()


def get_torch_dtype(torch_dtype):
    if not isinstance(torch_dtype, str):
        return torch_dtype
    return {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[torch_dtype]


def if_freeze_params(module, freeze):
    if isinstance(module, list):
        return [if_freeze_params(m, freeze) for m in module]
    elif isinstance(module, dict):
        return {k: if_freeze_params(v, freeze) for k, v in module.items()}
    else:
        getattr(module, "train" if not freeze else "eval")()
        for param in module.parameters():
            param.requires_grad = not freeze
        return module
        
        
def unfreeze_params(module):
    return if_freeze_params(module, False)


def freeze_params(module):
    return if_freeze_params(module, True)
        
        
def freeze_partial_embeddings(module, unfreeze_length):
    unfreeze_params(module)
    def _freeze_partial_embeddings(grad, unfreeze_length=unfreeze_length):
        grad[:unfreeze_length].zero_()
        return grad
    module.weight.register_hook(_freeze_partial_embeddings)
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.register_hook(_freeze_partial_embeddings)


# def monkey_patch_of_evaluate_without_collect_results_for_mmengine():
#     from mmengine.evaluator import BaseMetric
    
#     def evaluate(_self, size: int) -> dict:
#         if len(_self.results) == 0:
#             print_log(
#                 f'{_self.__class__.__name__} got empty `self.results`. Please '
#                 'ensure that the processed results are properly added into '
#                 '`self.results` in `process` method.',
#                 logger='current',
#                 level=logging.WARNING)
            
#         results = _self.results

#         # cast all tensors in results list to cpu
#         results = _to_cpu(results)
#         _metrics = _self.compute_metrics(results)  # type: ignore
#         # Add prefix to metric names
#         if _self.prefix:
#             _metrics = {
#                 '/'.join((_self.prefix, k)): v
#                 for k, v in _metrics.items()
#             }
#         metrics = [_metrics]

#         # reset the results list
#         _self.results.clear()
#         return metrics[0]
