LOWER_LIST=ops_bf16.txt python scripts/gaudi_spawn.py --world_size 8 --use_deepspeed scripts/train.py \
--dataset_path "./dataset" \
--model_path /mnt/weka/data/mlperf_models/llama-v2-fused-qkv \
--max_seq_len 8192 \
--bf16 True \
--logging_steps 24 \
--eval_steps 48 \
--save_steps 999 \
--output_dir "/tmp/llama-70b" \
--dataloader_num_workers 4 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--lr_scheduler_type "cosine" \
--learning_rate 4e-4 \
--weight_decay 0.0001 \
--warmup_ratio 0 \
--max_grad_norm 0.3 \
--use_gradient_checkpointing True \
--target_eval_loss 0.925 \
--use_peft_lora True \
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.1 \
--max_steps 1024 \
--seed 42 \
--use_habana \
--use_lazy_mode \
--deepspeed configs/ds_zero3.json \
--evaluation_strategy steps \
--gradient_checkpointing True \
--cache_dir /tmp/datasets \
--warmup \
--adjust_throughput \
--deterministic \
--flash_attention_fast_softmax_enable true \
--flash_attention_recompute_enable false \
--lora_target_modules "qkv_proj,o_proj"
