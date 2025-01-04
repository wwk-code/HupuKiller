# ref
dataset="amazon_reviews_multi"
subset="en"
python run_classification.py \
    --model_name_or_path  google-bert/bert-base-uncased \
    --dataset_name ${dataset} \
    --dataset_config_name ${subset} \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name "review_title,review_body,product_category" \
    --text_column_delimiter "\n" \
    --label_column_name stars \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir /tmp/${dataset}_${subset}/



# For Training Aruguments:
python run_classification.py \
    --model_name_or_path  /root/.cache/huggingface/hub/models--google-bert--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f \
    --train_file /data/workspace/projects/HupuKiller/assets/distilBert/datas/train.json \
    --validation_file /data/workspace/projects/HupuKiller/assets/distilBert/datas/evaluation.json \
    --text_column_names text \
    --label_column_name label \
    --max_seq_length 256 \
    --output_dir /data/workspace/projects/HupuKiller/outputs/DistilBert/fineTune \
    --overwrite_output_dir True \
    --do_train True \
    --do_eval True \
    --eval_strategy epoch \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --log_level info \
    --logging_dir /data/workspace/projects/HupuKiller/outputs/DistilBert/fineTune \
    --logging_strategy steps \
    --logging_steps 100 \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --bf16 True \
    --tf32 Ture  \
    --dataloader_num_workers 1 \
    --load_best_model_at_end True \
    --preprocessing_num_workers 1 \
    --shuffle_train_dataset True 



    



