# Common Notes

0. 修改源码的文件记录
   ```bash
   1. transformers/models/bert/modeling_bert.py: 
   1.1 自定义了: custom_index_1_axis_for_2_dim_obj_operation, 在文件中有些索引操作语句进行了替换,直接调用它,以支持Proxy类型变量
   1.2 torch.ao.quantization.observer.py的HistogramObserver.forward方法中新添加了:
           # wdhs's adding
           if isinstance(x_orig,torch.Size):
               return x_orig
   ```


```bash
报错:
(HupuKiller) root@DESKTOP-9RQB5NI:/data/workspace/projects/HupuKiller/src/py/distilBert#  cd /data/workspace/projects/HupuKiller/src/py/distilBert ; /usr/bin/env /data/workspace/anaconda/envs/HupuKiller/bin/python /root/.vscode-server/extensions/ms-python.debugpy-2024.14.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 53833 -- /data/workspace/projects/HupuKiller/src/py/distilBert/distilBertInfer.py 
/data/workspace/projects/HupuKiller/src/py/distilBert/distilBertInfer.py:140: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  customDistilBert = torch.load(checkpointFilePath).to(device)
/data/workspace/projects/HupuKiller/local_libs/site-packages/torch/_utils.py:413: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  device=storage.device,
Traceback (most recent call last):
  File "/data/workspace/projects/HupuKiller/local_libs/site-packages/torch/fx/graph_module.py", line 348, in __call__
    return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
  File "/data/workspace/projects/HupuKiller/local_libs/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/workspace/projects/HupuKiller/local_libs/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "<eval_with_key>.1 from <eval_with_key>.0:12 in forward", line 52, in forward
    quantize_per_tensor_4 = torch.quantize_per_tensor(add_3, encoder_0_attention_self_scale_0, encoder_0_attention_self_zero_point_0, torch.quint8);  add_3 = encoder_0_attention_self_scale_0 = encoder_0_attention_self_zero_point_0 = None
TypeError: expected Tensor as element 0 in argument 0, but got int

Call using an FX-traced Module, line 52 of the traced Module's generated forward function:
    encoder_0_attention_self_zero_point_0 = self.encoder_0_attention_self_zero_point_0
    quantize_per_tensor_4 = torch.quantize_per_tensor(add_3, encoder_0_attention_self_scale_0, encoder_0_attention_self_zero_point_0, torch.quint8);  add_3 = encoder_0_attention_self_scale_0 = encoder_0_attention_self_zero_point_0 = None
```
