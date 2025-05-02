import os
import torch
import pandas as pd
import torch.distributed as dist


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    resume = config.MODEL.RESUME
    logger.info(f"==============> Resuming form {resume}....................")

    checkpoint = torch.load(resume, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    std_rmse = torch.inf
    if 'std_rmse' in checkpoint:
        std_rmse = checkpoint['std_rmse']
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{resume}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()
    return std_rmse

def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} ......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']
    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()

def save_checkpoint(config, epoch, model, std_rmse, optimizer, lr_scheduler, loss_scaler, logger):
    save_path = os.path.join(config.MODEL.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'std_rmse': std_rmse,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

class SavingTool:

    def __init__(self):
        self.min_std_rmse = torch.inf
    def __call__(self, config, epoch, model, std_rmse, optimizer, lr_scheduler, loss_scaler, logger):

        logger.info(f'Current std_rmse: {std_rmse:.6f} versus min_std_rmse: {self.min_std_rmse:.6f}')

        if std_rmse < self.min_std_rmse:
            logger.info(f'A new land!')
            self.min_std_rmse = std_rmse
        else:
            logger.info(f'It seems nothing happens.')
        
        self.save_checkpoint(config, epoch, model, std_rmse, optimizer, lr_scheduler, loss_scaler, logger)
        logger.info(f'min_std_rmse: {self.min_std_rmse:.6f}')

    def save_checkpoint(self, config, epoch, model, std_rmse, optimizer, lr_scheduler, loss_scaler, logger):
        save_path = os.path.join(config.MODEL.OUTPUT, f'ckpt_epoch_{epoch}.pth')
        save_state = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'std_rmse': std_rmse,
                    'scaler': loss_scaler.state_dict(),
                    'epoch': epoch,
                    'config': config}
        
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def beautiful_metrics(config, rmse, std_rmse, acc, step):

    std_rmse = std_rmse.tolist()
    std_rmse_frame = pd.DataFrame(std_rmse, columns=['t2m', 't', 'z', 'u10', 'v10', 'mean'], index=[6 * i for i in range(1, 1 + step)])
    std_rmse_frame = std_rmse_frame.applymap(lambda x:('%.4f')%x)

    rmse = rmse.tolist()
    rmse_frame = pd.DataFrame(rmse, columns=['t2m', 't', 'z', 'u10', 'v10'], index=[6 * i for i in range(1, 1 + step)])
    rmse_frame = rmse_frame.applymap(lambda x:('%.4f')%x)

    acc = acc.tolist()
    acc_frame = pd.DataFrame(acc, columns=['t2m', 't', 'z', 'u10', 'v10'], index=[6 * i for i in range(1, 1 + step)])
    acc_frame = acc_frame.applymap(lambda x:('%.4f')%x)

    return rmse_frame, std_rmse_frame, acc_frame

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

class Tensor_AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, size_):
        self.size_ = size_
        self.reset()

    def reset(self):
        self.val = torch.zeros(self.size_)
        self.avg = torch.zeros(self.size_)
        self.sum = torch.zeros(self.size_)
        self.count = torch.zeros(self.size_)

    def update(self, val):
        isFinite = torch.isfinite(val)  # (T, B, 5)
        isFinite_ = isFinite.sum(dim=1) # (T, 5)
        finite_val = torch.where(isFinite, val, 0).sum(dim=1)   # (T, 5)
        self.val = finite_val / isFinite_
        self.sum += finite_val
        self.count += isFinite_
        self.avg = self.sum / self.count

def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == 'inf':
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place

                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                
                norm = ampscaler_get_grad_norm(parameters)

            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
