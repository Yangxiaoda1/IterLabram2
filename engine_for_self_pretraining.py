# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

from cgitb import enable
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from einops import rearrange
from contextlib import nullcontext

def random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # mask = np.hstack([
        #     np.zeros(len_keep),
        #     np.ones(L - len_keep),
        # ])
        # np.random.shuffle(mask)

        return mask.to(torch.bool)


def train_one_epoch(model: torch.nn.Module, vqnsp: torch.nn.Module, gtmodel: torch.nn.Module,
                    data_loader_list: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, ch_names_list=None, args=None):
    model.train()
    gtmodel.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss()

    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        if len(data_loader) == 0:
            continue
        input_chans = utils.get_input_chans(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
            # assign learning rate & weight decay for each step
            it = start_steps + step + step_loader  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            
            samples = batch
            # from IPython import embed; embed()
            samples = samples.float().to(device, non_blocking=True) / 100
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
            # 随机mask
            bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=0.2).to(device, non_blocking=True)
            # from IPython import embed; embed()

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    input_ids = vqnsp.get_codebook_indices(samples, input_chans)

                # 只获取被mask位置的标签
                labels = input_ids[bool_masked_pos]
                # 不再获取未被mask位置的标签
                # labels_sym 只是一个占位符，保持接口兼容性
                labels_sym = torch.zeros_like(labels)

            my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
            with my_context():
                with torch.cuda.amp.autocast(): # enabled=False
                    outputs = model(samples, input_chans, bool_masked_pos=bool_masked_pos)
                    with torch.no_grad():
                        # 只获取被mask位置的预测
                        gtoutputs, _ = gtmodel(samples, input_chans, bool_masked_pos=bool_masked_pos)
                    
                    # 软标签蒸馏 - 使用温度缩放的软标签
                    temperature = 2.0  # 温度系数，控制分布的平滑度
                    
                    x_rec, x_rec_sym = outputs
                    if args.supervisemode=="transformer":
                        # 使用软标签和KL散度
                        soft_targets = F.softmax(gtoutputs / temperature, dim=1)
                        log_probs = F.log_softmax(x_rec / temperature, dim=1)
                        loss_rec = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
                        # 第二次mask的损失设为0
                        loss_rec_sym = torch.tensor(0.0).to(device)
                    elif args.supervisemode=="vq":
                        # 原始的硬标签方式
                        loss_rec = loss_fn(x_rec, labels)
                        # 第二次mask的损失设为0
                        loss_rec_sym = torch.tensor(0.0).to(device)
                    
                    # 总损失只包含第一次mask的损失
                    loss = loss_rec

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
                
                sys.exit(1)

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= args.gradient_accumulation_steps
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order, update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            
            # 计算准确率
            with torch.no_grad():
                if args.supervisemode=="transformer":
                    # 使用教师模型的硬标签预测来计算准确率
                    teacher_predict = gtoutputs.max(-1)[1]
                    mlm_acc = (x_rec.max(-1)[1] == teacher_predict).float().mean().item()
                else:
                    # vq模式下使用原始标签计算准确率
                    mlm_acc = (x_rec.max(-1)[1] == labels).float().mean().item()
                
                # 第二次mask的准确率设为0或随机值，仅保持接口兼容性
                mlm_acc_sym = 0.0
            metric_logger.update(mlm_acc=mlm_acc)
            metric_logger.update(mlm_acc_sym=mlm_acc_sym)
            metric_logger.update(loss_rec=loss_rec.item())

            if log_writer is not None:
                log_writer.update(mlm_acc=mlm_acc, head="loss")
                log_writer.update(mlm_acc_sym=mlm_acc_sym, head="loss")
                log_writer.update(loss_rec=loss_rec.item(), head="loss")

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

