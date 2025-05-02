import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import get_config
from models import build_model
from logger import create_logger
from timm.utils import AverageMeter
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from data import build_loader, get_adjacency
from criterion import Criterion, Validation
from utils import load_checkpoint, load_pretrained, SavingTool, NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor, beautiful_metrics, Tensor_AverageMeter

def parse_option():
    parser = argparse.ArgumentParser('Model training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',)


    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--pretrained', help='pretrained from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main(config):
    
    local_rank = int(os.environ["RANK"])
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(config, local_rank)
    adj_mat, edge_mat = get_adjacency(config.MODEL.KERNEL_ALPHA, 5)
    adj_mat, edge_mat = adj_mat.to_sparse_csr().half().cuda(), edge_mat.to_sparse_csr().half().cuda()

    config.defrost()
    config.DATA.DATAMEAN = [torch.load('./Storages/DataStat/dataMean', map_location=torch.device('cpu')).float().cuda()]
    config.DATA.DATASTD = [torch.load('./Storages/DataStat/dataStd', map_location=torch.device('cpu')).float().cuda()]
    config.DATA.LAT_LON_MESH = [torch.load('./Storages/lat_lon_mesh', map_location=torch.device('cpu')).float().cuda()]
    config.DATA.DATACLIM = [torch.load('./Storages/DataStat/dataClim', map_location=torch.device('cpu')).float().cuda()]
    config.DATA.CONSTANTS = [torch.load('./Storages/constants', map_location=torch.device('cpu')).float().cuda()]
    config.freeze()

    logger.info(f"Creating model:{config.MODEL.TYPE}")
    model = build_model(config, adj_mat, edge_mat)
    logger.info(str(model)) 

    model_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of model params: {model_n_parameters}")
    
    model.cuda()
    model_without_ddp = model
    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, 
                                                      broadcast_buffers=False, find_unused_parameters=True)
    loss_scaler = NativeScalerWithGradNormCount()
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion = Criterion(config)
    validation = Validation(config)
    savingTool = SavingTool()

    if config.TRAIN.AUTO_RESUME:
        model_resume_file = auto_resume_helper(config.MODEL.OUTPUT)
        if model_resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {model_resume_file}")
            config.defrost()
            config.MODEL.RESUME = model_resume_file
            config.freeze()
            logger.info(f'auto resuming from {model_resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.MODEL.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        std_rmse = load_checkpoint(config, model_without_ddp, optimizer, 
                                   lr_scheduler, loss_scaler, logger)
        logger.info(f'recorded std_rmse of checkpoint: {std_rmse}')

        rmse, std_rmse, acc = validate(config, validation, data_loader_val, model, step=config.TRAIN_STEP)
        rmse_frame, std_rmse_frame, acc_frame = beautiful_metrics(config, rmse, std_rmse, acc, step=config.TRAIN_STEP)
        logger.info(f"--------rmse of the network on the {len(dataset_val)} validation--------")
        logger.info(rmse_frame)

        mean_std_rmse = std_rmse[:, 5].mean()
        logger.info(f'true std_rmse of checkpoint: {mean_std_rmse}')
        savingTool.min_std_rmse = mean_std_rmse

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        rmse, std_rmse, acc = validate(config, validation, data_loader_val, model, step=config.TRAIN_STEP)
        rmse_frame, std_rmse_frame, acc_frame = beautiful_metrics(config, rmse, std_rmse, acc, step=config.TRAIN_STEP)
        logger.info(f"--------rmse of the network on the {len(dataset_val)} validation--------")
        logger.info(rmse_frame)

        mean_std_rmse = std_rmse[:, 5].mean()
        logger.info(f'true std_rmse of checkpoint: {mean_std_rmse}')
        savingTool.min_std_rmse = mean_std_rmse
        
    if config.EVAL_MODE:
        rmse, std_rmse, acc = validate(config, validation, data_loader_test, model, step=config.EVAL_STEP)
        
        rmse_frame, std_rmse_frame, acc_frame = beautiful_metrics(config, rmse, std_rmse, acc, step=config.EVAL_STEP)
        logger.info(f"--------rmse of the network on the {len(dataset_test)} test--------")
        logger.info(rmse_frame)

        logger.info(f"--------acc of the network on the {len(dataset_test)} test--------")
        logger.info(acc_frame)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, data_loader_train, model, criterion, optimizer, epoch, lr_scheduler, loss_scaler, step=config.TRAIN_STEP)

        rmse, std_rmse, acc = validate(config, validation, data_loader_val, model, step=config.TRAIN_STEP)
            
        rmse_frame, std_rmse_frame, acc_frame = beautiful_metrics(config, rmse, std_rmse, acc, step=config.TRAIN_STEP)
        
        logger.info(f"--------rmse of the network on the {len(dataset_val)} validation--------")
        logger.info(rmse_frame)

        mean_std_rmse = std_rmse[:, 5].mean()

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            savingTool(config, epoch, model_without_ddp, mean_std_rmse, optimizer, lr_scheduler, loss_scaler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, data_loader, model, criterion, optimizer, 
                    epoch, lr_scheduler, loss_scaler, step):
    
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    dist_meter = AverageMeter()
    distVel_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (dataStates, dataInfo, targetStates) in enumerate(data_loader):
        dataStates = dataStates.cuda(non_blocking=True) # (B, 5, 32, 64)
        targetStates = targetStates.transpose(0, 1).cuda(non_blocking=True)[0:step] # (B, T, 5, 32, 64) 2 (T, B, 5, 32, 64)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            predict_dataStates, predict_velocity = model(dataStates, step)
            loss, dist, distVel = criterion.forward(predict_dataStates, targetStates, predict_velocity)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
        
        norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD, parameters=model.parameters(), create_graph=False,
                            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)

        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targetStates.size(1))
        dist_meter.update(dist.item(), targetStates.size(1))
        if distVel: distVel_meter.update(distVel.item(), targetStates.size(1))
        if norm is not None:  # loss_scaler return None if not update
            norm_meter.update(norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.8f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.5f} ({loss_meter.avg:.5f})\t'
                f'dist {dist_meter.val:.5f} ({dist_meter.avg:.5f})\t'
                f'distVel {distVel_meter.val:.7f} ({distVel_meter.avg:.7f})\t'
                f'model_grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.5f} ({scaler_meter.avg:.5f})\t'
                f'mem {memory_used:.0f}MB\t'
                f'step:{step}')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(config, validation, data_loader, model, step):
    model.eval()

    batch_time = AverageMeter()
    rmse_meter = Tensor_AverageMeter([step, 5])
    acc_meter = Tensor_AverageMeter([step, 5])

    end = time.time()
    for idx, (dataStates, dataInfo, targetStates) in enumerate(data_loader):
        dataStates = dataStates.cuda(non_blocking=True) # (B, 5, 32, 64)
        targetStates = targetStates.transpose(0, 1).cuda(non_blocking=True)[0:step] # (B, T, 5, 32, 64) 2 (T, B, 5, 32, 64)

        predict_dataStates, predict_velocity = model(dataStates, step)

        # measure accuracy and record loss
        rmse = validation.compute_weighted_rmse(predict_dataStates, targetStates)
        acc = validation.compute_weighted_acc(predict_dataStates, targetStates)

        rmse = reduce_tensor(rmse)
        acc = reduce_tensor(acc)

        rmse_meter.update(rmse.cpu())
        acc_meter.update(acc.cpu())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'rmse {rmse_meter.val} ({rmse_meter.avg})\t'
                f'acc {acc_meter.val} ({acc_meter.avg})\t'
                f'Mem {memory_used:.0f}MB')
    
    std_rmse = rmse_meter.avg / config.DATA.DATASTD[0][None, :].cpu()
    mean_std_rmse = std_rmse.mean(dim=1)[:, None]
    std_rmse = torch.cat([std_rmse, mean_std_rmse], dim=1)

    logger.info(f' * rmse {rmse_meter.avg}\t'
                f' * std_rmse {std_rmse}\t'
                f' * acc {acc_meter.avg}\t'
                f' * count {rmse_meter.count}')
    return rmse_meter.avg, std_rmse, acc_meter.avg

if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    torch.distributed.init_process_group(backend='gloo')
    local_rank = int(os.environ["LOCAL_RANK"])
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    device = torch.device(device_id)

    torch.cuda.set_device(device)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.MODEL.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.MODEL.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.TYPE}")

    if dist.get_rank() == 0:
        path = os.path.join(config.MODEL.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)