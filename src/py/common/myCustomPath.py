import os,sys


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


base_model_path = "/root/.cache/huggingface/hub/models--shenzhi-wang--Llama3-8B-Chinese-Chat/snapshots/f25f13cb2571e70e285121faceac92926b51e6f5"  # 原始模型路径
nba_final_average_qa_lora_weights_path = "/data/workspace/projects/llamaLearn/LLaMA-Factory/saves/Llama-3-8B-Chinese-Chat/lora/HupuKillerNBAFinalAverageSFT_0"  # LoRA 输出权重的路径
# nba_final_average_qa_lora_weights_path = "/data/workspace/projects/llamaLearn/LLaMA-Factory/saves/Llama-3-8B-Chinese-Chat/lora/HupuKillerNBAFinalAverageSFT_abstract_and_concise_epoch30"  # LoRA 输出权重的路径
nba_final_average_qa_loraMerged_output_path = os.path.join(project_root_dir,'outputs','LoraMerged','NBAFinalAverageDatas')
sentence_transformer_path = '/root/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/8d6b950845285729817bf8e1af1861502c2fed0c/'



