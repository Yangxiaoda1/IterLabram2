# OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
#         --output_dir /data/home/yangxiaoda/LaBraM/checkpoints/labram_base_pretraining/finetune_tuab_epoch1pretraining \
#         --log_dir /data/home/yangxiaoda/LaBraM/log/labram_base_pretraining/finetune_tuab_epoch1pretraining \
#         --finetune /data/home/yangxiaoda/LaBraM/checkpoints/labram_base_pretraining/checkpoint-1.pth \
#         --weight_decay 0.05 \
#         --batch_size 64 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 5 \
#         --epochs 50 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --save_ckpt_freq 5 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --dataset TUAB \
#         --disable_qkv_bias \
#         --seed 0


python run_class_finetuning.py \
        --output_dir /data/home/yangxiaoda/LaBraM/checkpoints/labram_base_pretraining/finetune_tuab_epoch1pretraining \
        --log_dir /data/home/yangxiaoda/LaBraM/log/labram_base_pretraining/finetune_tuab_epoch1pretraining \
        --finetune /data/home/yangxiaoda/LaBraM/checkpoints/labram_base_pretraining/checkpoint-1.pth \
        --weight_decay 0.05 \
        --batch_size 256 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 5 \
        --epochs 30     \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --save_ckpt_freq 5 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset TUAB \
        --disable_qkv_bias \
        --seed 0