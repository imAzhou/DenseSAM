import os
import torch
import argparse
from densesam.utils import set_seed, get_logger, get_train_strategy
from densesam.models.dense_sam import DenseSAMNet
from densesam.datasets.create_loader import gene_loader_trainval
from mmengine.config import Config
from densesam.utils.metrics import get_metrics
from scripts.main import train_one_epoch,val_one_epoch

parser = argparse.ArgumentParser()

# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()


def main(logger_name, cfg):
    # load datasets
    train_dataloader, val_dataloader, metainfo, restinfo = gene_loader_trainval(
        dataset_config = d_cfg, seed = args.seed)

    # register model
    model = DenseSAMNet(
        use_local = cfg.use_local,
        use_global = cfg.use_global,
        use_boundary_head = cfg.use_boundary_head,
        use_embed = cfg.dataset.load_embed,
        sam_ckpt = cfg.sam_ckpt,
        sam_type = cfg.sam_type,
        device = device,
        use_inner_idx = cfg.use_inner_idx
    ).to(device)
    
    # create logger
    logger,files_save_dir = get_logger(
        args.record_save_dir, model, cfg, logger_name)
    pth_save_dir = f'{files_save_dir}/checkpoints'
    # get optimizer and lr_scheduler
    optimizer,lr_scheduler = get_train_strategy(model, cfg)
    # get evaluator
    evaluators = get_metrics(cfg.metrics, metainfo, restinfo, cfg.inst_indicator)

    # train and val in each epoch
    all_metrics,all_value = [],[]
    max_value,max_epoch = 0,0
    for epoch_num in range(cfg.max_epochs):
        # train
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"epoch: {epoch_num}, learning rate: {current_lr:.6f}")
        
        train_one_epoch(
            model, train_dataloader, optimizer, logger, device, cfg, args.print_interval)
        
        lr_scheduler.step()
        if cfg.save_each_epoch:
            save_mode_path = os.path.join(pth_save_dir, 'epoch_' + str(epoch_num) + '.pth')
            model.save_parameters(save_mode_path)
            logger.info(f"save model to {save_mode_path}")

        # val
        metrics = val_one_epoch(model, val_dataloader, evaluators, device, cfg)
        best_score = -1
        if cfg.best_score_index == 'pixel':
            best_score = metrics['semantic']['mIoU']
        if cfg.best_score_index == 'panoptic':
            best_score = metrics['panoptic']['coco_panoptic/PQ']
        if cfg.best_score_index == 'inst':
            index = cfg.inst_indicator
            if cfg.inst_indicator == 'all':
                index = 'PQ'
            best_score = metrics['inst'][index]
            
        if best_score > max_value:
            max_value = best_score
            max_epoch = epoch_num
            save_mode_path = os.path.join(pth_save_dir, 'best.pth')
            model.save_parameters(save_mode_path)

        all_value.append(best_score)
        all_metrics.append(f'epoch: {epoch_num}' + str(metrics) + '\n')
    
    print(f'max_value: {max_value}, max_epoch: {max_epoch}')
    # save result file
    config_file = os.path.join(files_save_dir, 'pred_result.txt')
    with open(config_file, 'w') as f:
        f.writelines(all_metrics)
        f.write(f'\nmax_value: {max_value}, max_epoch: {max_epoch}\n')
        f.write(str(all_value))


if __name__ == "__main__":
    device = torch.device(args.device)

    d_cfg = Config.fromfile(args.dataset_config_file)
    model_strategy_config_file = 'densesam/configs/model_strategy.py'
    m_s_cfg = Config.fromfile(model_strategy_config_file)

    cfg = Config()
    for sub_cfg in [d_cfg, m_s_cfg]:
        cfg.merge_from_dict(sub_cfg.to_dict())
    
    search_lr = cfg.get('search_lr', None)
    for idx,base_lr in enumerate(search_lr):
        cfg.base_lr = base_lr
        set_seed(args.seed)
        logger_name = f'lr_{idx}'
        main(logger_name, cfg)



'''
python scripts/train.py \
    densesam/configs/datasets/conic.py \
    --record_save_dir logs/conic \
    --print_interval 20 \
    --device cuda:1
    
MedicalDatasets/CoNIC/train/img_dir/crag_15_0023.png
'''
