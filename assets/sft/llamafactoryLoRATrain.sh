
# all-Modules LoRA fine-tune
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /root/.cache/huggingface/hub/models--shenzhi-wang--Llama3-8B-Chinese-Chat/snapshots/f25f13cb2571e70e285121faceac92926b51e6f5 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset HupuKiller_NBAFinalAverageDatas_Abstract,HupuKiller_NBAFinalAverageDatas_Concise \
    --cutoff_len 256 \
    --learning_rate 5e-05 \
    --num_train_epochs 20.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Llama-3-8B-Chinese-Chat/lora/HupuKillerNBAFinalAverageSFT_0 \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target all \
    --overwrite_output_dir \
    --local_rank 0



# specific-Modules LoRA fine-tune
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /root/.cache/huggingface/hub/models--shenzhi-wang--Llama3-8B-Chinese-Chat/snapshots/f25f13cb2571e70e285121faceac92926b51e6f5 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset HupuKiller_NBAFinalAverageDatas \
    --cutoff_len 256 \
    --learning_rate 5e-05 \
    --num_train_epochs 50.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Llama-3-8B-Chinese-Chat/lora/HupuKillerNBAFinalAverageSFT_0 \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target "q_proj,k_proj,v_proj,gate_proj" \
    --overwrite_output_dir \
    --local_rank 0
    



# saves/Llama-3-8B-Chinese-Chat/lora/train_2024-12-10-22-15-42