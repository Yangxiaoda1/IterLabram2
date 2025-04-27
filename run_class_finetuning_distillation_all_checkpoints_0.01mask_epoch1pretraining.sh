#!/bin/bash

# 定义要处理的检查点文件
CHECKPOINTS=(
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-0.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-1.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-2.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-3.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-4.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-5.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-6.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-7.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-8.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-9.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-10.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-11.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-12.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-13.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-14.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-15.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-16.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-17.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-18.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-19.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-20.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-21.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-22.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-23.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-24.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-25.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-26.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-27.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-28.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint-29.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_0.01mask_epoch1pretraining/checkpoint.pth"
    
)

# 确保没有遗留的torchrun进程
pkill -f "torchrun"
sleep 5  # 等待进程完全终止

# 遍历所有检查点文件
for checkpoint in "${CHECKPOINTS[@]}"; do
    # 提取文件名
    filename=$(basename "$checkpoint")
    
    # 检查是否是需要跳过的检查点
    skip=false
    for skip_ckpt in "${SKIP_CHECKPOINTS[@]}"; do
        if [[ "$filename" == "$skip_ckpt" ]]; then
            skip=true
            break
        fi
    done
    
    # 如果需要跳过，则继续下一个循环
    if $skip; then
        echo "跳过 $filename"
        continue
    fi
    
    # 提取检查点编号（用于输出目录和日志目录）
    ckpt_num=$(echo $filename | sed 's/checkpoint-\(.*\)\.pth/\1/')
    
    echo "开始微调 $filename"
    
    # 运行微调命令
    OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
        --output_dir ./checkpoints/labram_distillation1_0.01mask_epoch1pretraining/finetunetuab${ckpt_num}/ \
        --log_dir ./log/labram_distillation1_0.01mask_epoch1pretraining/finetunetuab${ckpt_num} \
        --model labram_base_patch200_200 \
        --finetune $checkpoint \
        --weight_decay 0.05 \
        --batch_size 256 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 3 \
        --epochs 30 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --dist_eval \
        --save_ckpt_freq 5 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset TUAB \
        --disable_qkv_bias \
        --seed 0
    
    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
        echo "微调 $filename 失败，退出代码: $?"
        # 确保所有相关进程都被终止
        pkill -f "torchrun"
        sleep 5  # 等待进程完全终止
    else
        echo "完成微调 $filename"
    fi
    
    # 额外确保没有遗留的torchrun进程
    pkill -f "torchrun"
    sleep 5  # 等待进程完全终止
done

echo "所有检查点微调完成！" 