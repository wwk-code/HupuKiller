import os,sys,torch,time

# 将本地包路径添加到 sys.path 的最前面
local_packages_path = os.path.abspath("/data/workspace/projects/HupuKiller/local_libs/site-packages")
sys.path.insert(0, local_packages_path)

import torch
from torch.ao.quantization import get_default_qconfig,default_qconfig,default_weight_only_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import  QConfig,QConfigMapping,default_dynamic_qconfig
from transformers import pipeline,AutoTokenizer
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch.ao.quantization.observer import HistogramObserver,default_observer,default_weight_observer

# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.getcwd())
sys.path.append(py_src_root_dir)

from common.myFile import * 
from customDistilBert import CustomDistilBert 

from distilBertInfer import getDataLoaderAndLabels


# 自定义 HistogramObserver (目前的计算图IR中存在将torch.Size作为HistogramObserver输入的情况，需要特殊处理)
class CustomHistogramObserver(HistogramObserver):
    def forward(self, x_orig):
        # 如果输入是 torch.Size，直接返回,不进行直方图信息统计
        if isinstance(x_orig, torch.Size):
            return x_orig
        else: 
            return super().forward(x_orig)

customObserver = CustomHistogramObserver.with_args(dtype=torch.qint8,reduce_range=True) 

# custom_qconfig = QConfig(
#     activation = customObserver,
#     weight=default_weight_observer
# )

# custom_qconfig = default_dynamic_qconfig
custom_qconfig = get_default_qconfig("x86")

def do_fx_mode_ptq():

    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        device = torch.device('cpu')
        model_path = '/data/workspace/projects/HupuKiller/outputs/DistilBert/fineTune'
        bert_base = BertForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        customDistilBert = CustomDistilBert(bert_base).to(device)
        checkpointFilePath = '/data/workspace/projects/HupuKiller/outputs/DistilBert/distillation/model_epoch_4.pth'
        distilCheckpoints = torch.load(checkpointFilePath,map_location=device)
        customDistilBert.load_state_dict(distilCheckpoints)

        model = customDistilBert
        model.eval()

        dataLoader,rawLabels = getDataLoaderAndLabels(batchSize = 8,tokenizer=tokenizer,device=device)
        qconfig_mapping = QConfigMapping().set_global(custom_qconfig)

        # 打印 CUDA 操作日志
        # print(prof.key_averages().table(sort_by="cuda_time_total"))

        def calibrate(model: torch.nn.Module, data_loader: DataLoader):
            model.eval()
            with torch.no_grad():
                for batch in dataLoader:
                    batch_input_ids, batch_attention_mask = batch

                    # 新添加逻辑,为了可以通过 torch.ao.quantization.HistogramObserver 的观测
                    batch_attention_mask = batch_attention_mask.float()  # 为了解决某个报错

                    model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

        example_inputs = (next(iter(dataLoader))[0]) # get an example input
        prepared_model = prepare_fx(model, qconfig_mapping,example_inputs)  # fuse modules and insert observers
        calibrate(prepared_model, dataLoader)  # run calibration on sample data
        quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model


    # 这段代码存在问题，会遭遇报错: AttributeError: 'Node' object has no attribute 'size'
    #     for node in quantized_model.graph.nodes:
    #         if node.target == 'size':
    #             with quantized_model.graph.inserting_after(node):
    #                 new_node = quantized_model.graph.call_function(torch.tensor, args=(node.args[0].size(),))
    #                 node.replace_all_uses_with(new_node)
    #             quantized_model.graph.erase_node(node)

    # # 重新编译模型
    # quantized_model.recompile()
    
    temp = 1
        
    return quantized_model


def saveQuantModelStates(model):
    ptq_output_path = '/data/workspace/projects/HupuKiller/outputs/DistilBert/quant/ptq'
    # torch.save(model.state_dict(), f"{ptq_output_path}/ptq_checkpoints.pth")  # 保存PTQ权重
    torch.save(model, f"{ptq_output_path}/full_ptq_checkpoints.pth")  # 保存PTQ权重


if __name__ == '__main__':

    ptq_model = do_fx_mode_ptq()

    saveQuantModelStates(ptq_model)

