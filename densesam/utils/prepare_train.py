import os
import time
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from mmengine.logging import MMLogger
from densesam.utils import get_parameter_number
from mmengine.config import Config

def get_logger(record_save_dir, model, print_cfg: Config, logger_name):
    # set record files
    save_dir_date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    files_save_dir = f'{record_save_dir}/{save_dir_date}'
    os.makedirs(files_save_dir, exist_ok=True)
    pth_save_dir = f'{files_save_dir}/checkpoints'
    os.makedirs(pth_save_dir, exist_ok=True)
    # save config file
    config_file = os.path.join(files_save_dir, 'config.py')
    print_cfg.dump(config_file)
    # save log file
    logger = MMLogger.get_instance(logger_name, log_file=f'{files_save_dir}/result.log')
    parameter_cnt = get_parameter_number(model)
    logger.info(f'total params: {parameter_cnt}')
    logger.info(f'update params:')
    for name,parameters in model.named_parameters():
        if parameters.requires_grad:
            logger.info(name)
    return logger,files_save_dir

def get_train_strategy(model, args):
    '''slow start & fast decay'''
    def lr_lambda(epoch):
        if epoch < args.warmup_epoch:
            return (epoch + 1) / args.warmup_epoch  # warm up 阶段线性增加
        else:
            # [base_lr*(args.gamma ** (epoch-args.warmup_epoch + 1))]
            # [0.005 *(0.9**i) for i in range(1,31)]
            return args.gamma ** (epoch-args.warmup_epoch + 1)
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.base_lr, betas = (0.9, 0.999), eps = 1e-08, weight_decay=0)
    
    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

    return optimizer,lr_scheduler