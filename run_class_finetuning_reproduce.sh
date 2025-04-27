# OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
#         --output_dir ./checkpoints/finetune_reproduce_try2rd/ \
#         --log_dir ./log/finetune_reproduce_try2rd/ \
#         --model labram_base_patch200_200 \
#         --finetune ./checkpoints/labram-base.pth \
#         --weight_decay 0.05 \
#         --batch_size 256 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 5 \
#         --epochs 30 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --save_ckpt_freq 5 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --dataset TUAB \
#         --disable_qkv_bias \
#         --seed 0

python run_class_finetuning.py \
        --output_dir ./checkpoints/finetune_reproduce_try2rd/ \
        --log_dir ./log/finetune_reproduce_try2rd/ \
        --model labram_base_patch200_200 \
        --finetune ./checkpoints/labram-base.pth \
        --weight_decay 0.05 \
        --batch_size 256 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 5 \
        --epochs 30 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --save_ckpt_freq 5 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset TUAB \
        --disable_qkv_bias \
        --seed 0