import torch
from safetensors.torch import load_file

# 定义适配器文件路径
ref_adapter_path = "/data/workspace/projects/HupuKiller/outputs/DPO/DPO_NABFinalAverageDatas/reference_adapter/adapter_model.safetensors"
train_adapter_path = "/data/workspace/projects/HupuKiller/outputs/DPO/DPO_NABFinalAverageDatas/training_adapter/adapter_model.safetensors"

# 加载适配器权重
ref_adapter_weights = load_file(ref_adapter_path)
train_adapter_weights = load_file(train_adapter_path)

# 比较权重
for key in ref_adapter_weights.keys():
    if key in train_adapter_weights:
        ref_param = ref_adapter_weights[key]
        train_param = train_adapter_weights[key]
        diff = (ref_param - train_param).abs().mean().item()
        
        print(f"Parameter: {key}, Mean Absolute Difference: {diff}")
    else:
        print(f"Parameter {key} not found in train_adapter.")