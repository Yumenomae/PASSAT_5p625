import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.ENABLE_AMP = False
_C.AMP_ENABLE = True
_C.AMP_OPT_LEVEL = ''
_C.TAG = 'default'
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 10
_C.SEED = 0
_C.TRAIN_STEP = 4   # 24 hours
_C.EVAL_STEP = 24   # 120 hours
_C.EVAL_MODE = False
_C.LOCAL_RANK = -1
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.SIZE = [32, 64]
_C.DATA.BATCH_SIZE = 2 
_C.DATA.ORIGINAL_DATA_PATH = './RawData'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8
_C.DATA.WINDOWSIZE = 5

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = ''
_C.MODEL.PRETRAINED = ''
_C.MODEL.KERNEL_ALPHA = 1.0
_C.MODEL.RESUME = ''
_C.MODEL.OUTPUT = ''
_C.MODEL.MAX_STEP = None
_C.MODEL.LAYERNORM = False


# -----------------------------------------------------------------------------
# PASSAT settings
# -----------------------------------------------------------------------------
_C.MODEL.PASSAT = CN()
_C.MODEL.PASSAT.EMBED_DIM = None
_C.MODEL.PASSAT.BACKBONE_DEPTHS = None
_C.MODEL.PASSAT.BRANCH_DEPTHS = None
_C.MODEL.PASSAT.LAMBDA_VELOCITY_VALUE = None
_C.MODEL.PASSAT.LAMBDA_VELOCITY_GRAD = 1

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 50
_C.TRAIN.WARMUP_EPOCHS = 1
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 1e-3
_C.TRAIN.WARMUP_LR = 0
_C.TRAIN.MIN_LR = 3e-7

# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 1.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Whether to use gradient checkpointing to save memory
_C.TRAIN.ACCUMULATION_STEPS = 1
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
        config.TRAIN.AUTO_RESUME = False
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
        config.TRAIN.AUTO_RESUME = False
    if _check_args('eval'):
        config.EVAL_MODE = True
        config.DATA.WINDOWSIZE = config.EVAL_STEP + 1
        config.DATA.BATCH_SIZE = 8

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config

