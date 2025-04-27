#!/bin/bash

# 定义要处理的检查点文件
CHECKPOINTS=(
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-0.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-1.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-2.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-3.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-4.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-5.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-6.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-7.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-8.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-9.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-10.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-11.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-12.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-13.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-14.pth"
    "/data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/checkpoint-15.pth"
    
    
)

# # 确保没有遗留的torchrun进程
# pkill -f "torchrun"
# sleep 5  # 等待进程完全终止

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
    python run_class_finetuning.py \
        --output_dir /data/home/yangxiaoda/LaBraM/checkpoints/labram_distillation1_softlabel_0.2mask/finetunetuab${ckpt_num}/ \
        --log_dir /data/home/yangxiaoda/LaBraM/log/labram_distillation1_softlabel_0.2mask/finetunetuab${ckpt_num} \
        --model labram_base_patch200_200 \
        --finetune $checkpoint \
        --weight_decay 0.05 \
        --batch_size 256 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 3 \
        --epochs 5 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --dist_eval \
        --save_ckpt_freq 1 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset TUAB \
        --disable_qkv_bias \
        --seed 0
    
    # 检查上一个命令的退出状态 
    if [ $? -ne 0 ]; then
        echo "微调 $filename 失败，退出代码: $?"
        # 获取当前脚本的PID
        SCRIPT_PID=$$
        # 只终止与当前脚本相关的进程
        for pid in $(ps -ef | grep "run_class_finetuning.py" | grep -v grep | awk '{print $2}'); do
            if ps -o ppid= -p $pid | grep -q $SCRIPT_PID; then
                kill $pid
                echo "终止进程 $pid"
            fi
        done
        sleep 5  # 等待进程完全终止
    else
        echo "成功完成微调 $filename"
        # 记录成功的检查点到日志文件
        echo "$filename 微调成功" >> /data/home/yangxiaoda/LaBraM/finetune_success.log
    fi
    
    # 确保没有与当前脚本相关的遗留进程
    SCRIPT_PID=$$
    for pid in $(ps -ef | grep "run_class_finetuning.py" | grep -v grep | awk '{print $2}'); do
        if ps -o ppid= -p $pid | grep -q $SCRIPT_PID; then
            kill $pid
            echo "终止遗留进程 $pid"
        fi
    done
    sleep 3  # 短暂等待确保进程终止
done

echo "所有检查点微调完成！" 