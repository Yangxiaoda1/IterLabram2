OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=1 run_labram_self_pretraining.py \
        --output_dir ./checkpoints/labram_distillation1_softlabel_0.2mask \
        --log_dir ./log/labram_distillation1_softlabel_0.2mask \
        --model labram_base_patch200_1600_8k_vocab \
        --gtmodel labram_base_patch200_1600_8k_vocab \
        --model_weight /data/home/yangxiaoda/LaBraM/checkpoints/labram-base.pth \
        --gtmodel_weight /data/home/yangxiaoda/LaBraM/checkpoints/labram-base.pth \
        --tokenizer_model vqnsp_encoder_base_decoder_3x200x12 \
        --tokenizer_weight /data/home/yangxiaoda/LaBraM/checkpoints/vqnsp.pth \
        --batch_size 64 \
        --lr 5e-4 \
        --warmup_epochs 5 \
        --clip_grad 3.0 \
        --drop_path 0. \
        --layer_scale_init_value 0.1 \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-8  \
        --epochs 10 \
        --save_ckpt_freq 1 \
        --codebook_dim 64 \
        --gradient_accumulation_steps 1 \
        --backbone transformer \
        --supervisemode transformer


# python run_labram_self_pretraining.py \
#         --output_dir ./checkpoints/labram_distillation1_softlabel_0.2mask \
#         --log_dir ./log/labram_distillation1_softlabel_0.2mask \
#         --model labram_base_patch200_1600_8k_vocab \
#         --gtmodel labram_base_patch200_1600_8k_vocab \
#         --model_weight /data/home/yangxiaoda/LaBraM/checkpoints/labram-base.pth \
#         --gtmodel_weight /data/home/yangxiaoda/LaBraM/checkpoints/labram-base.pth \
#         --tokenizer_model vqnsp_encoder_base_decoder_3x200x12 \
#         --tokenizer_weight /data/home/yangxiaoda/LaBraM/checkpoints/vqnsp.pth \
#         --batch_size 64 \
#         --lr 5e-4 \
#         --warmup_epochs 5 \
#         --clip_grad 3.0 \
#         --drop_path 0. \
#         --layer_scale_init_value 0.1 \
#         --opt_betas 0.9 0.98 \
#         --opt_eps 1e-8  \
#         --epochs 10 \
#         --save_ckpt_freq 1 \
#         --codebook_dim 64 \
#         --gradient_accumulation_steps 1 \
#         --backbone transformer \
#         --supervisemode transformer