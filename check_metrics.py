# 无意义，因为未微调过的model输出层和TUAB任务是不匹配的，

import torch
import numpy as np
from timm.models import create_model
import utils
import modeling_finetune
from einops import rearrange

def check_model_performance():
    # 加载模型
    model_name = 'labram_base_patch200_200'
    model_path = './checkpoints/labram-base.pth'
    
    # 设置参数
    num_classes = 1
    rel_pos_bias = False
    abs_pos_emb = True
    qkv_bias = False
    
    # 创建模型
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=rel_pos_bias,
        use_abs_pos_emb=abs_pos_emb,
        init_values=0.1,
        qkv_bias=qkv_bias,
    )
    
    # 加载预训练权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model_dict = model.state_dict()
    
    # 过滤权重
    pretrained_dict = {}
    for k, v in checkpoint['model'].items():
        if k in model_dict and model_dict[k].shape == v.shape:
            pretrained_dict[k] = v
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 加载TUAB数据集
    train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset(
        "/data/home/yangxiaoda/LaBraM/EEGdata/TUAB/isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf/processed"
    )
    
    # 获取通道名称
    ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF',
                'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
    input_chans = utils.get_input_chans(ch_names)
    
    # 创建数据加载器
    batch_size = 64
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )
    
    # 评估指标
    metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    
    # 评估模型
    pred_all = []
    target_all = []
    
    with torch.no_grad():
        for batch in test_loader:
            EEG = batch[0]
            target = batch[-1]
            
            EEG = EEG.float().to(device) / 100
            EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
            
            output = model(EEG, input_chans=input_chans)
            
            # 对于二分类任务
            target = target.float().to(device).unsqueeze(-1)
            output = torch.sigmoid(output).cpu()
            target = target.cpu()
            
            pred_all.append(output)
            target_all.append(target)
    
    pred_all = torch.cat(pred_all, dim=0).numpy()
    target_all = torch.cat(target_all, dim=0).numpy()
    
    # 计算指标
    results = utils.get_metrics(pred_all, target_all, metrics, is_binary=True)
    
    print("\n===== 模型在TUAB测试集上的性能 =====")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"AUC-PR: {results['pr_auc']:.4f}")
    print(f"AUROC: {results['roc_auc']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    
if __name__ == "__main__":
    check_model_performance() 