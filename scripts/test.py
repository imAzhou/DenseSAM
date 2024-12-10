import os
import torch
import argparse
from densesam.utils import set_seed
from densesam.models.dense_sam import DenseSAMNet
from densesam.datasets.create_loader import gene_loader_eval
from mmengine.config import Config
from densesam.utils.metrics import get_metrics
from scripts.main import val_one_epoch

parser = argparse.ArgumentParser()

# base args
parser.add_argument('config_file', type=str)
parser.add_argument('result_save_dir', type=str)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--visual_pred', action='store_true')
parser.add_argument('--draw_func', type=str)
parser.add_argument('--visual_interval', type=int, default=1)

args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    
    # load datasets
    val_dataloader, metainfo, restinfo = gene_loader_eval(
        dataset_config = cfg, seed = args.seed)

    # register model
    model = DenseSAMNet(
        sm_depth = cfg.semantic_module_depth,
        use_inner_feat = cfg.use_inner_feat,
        use_boundary_head = cfg.use_boundary_head,
        use_embed = cfg.dataset.load_embed,
        sam_ckpt = cfg.sam_ckpt,
        sam_type = cfg.sam_type,
        device = device,
        inter_idx = cfg.inter_idx
    ).to(device)
    
    visual_args = {'visual_pred':args.visual_pred}
    if args.visual_pred:
        pred_save_dir = f'{args.result_save_dir}/vis_pred'
        os.makedirs(pred_save_dir, exist_ok = True)
        visual_args['pred_save_dir'] = pred_save_dir
        visual_args['visual_interval'] = args.visual_interval
        visual_args['draw_func'] = args.draw_func

    # get evaluator
    evaluators = get_metrics(cfg.metrics, metainfo, restinfo, cfg.inst_indicator)

    # val
    model.load_parameters(args.ckpt_path)
    
    metric_results = val_one_epoch(
        model, val_dataloader, evaluators, device, cfg, visual_args)
    print(metric_results)


'''
draw_building_pred draw_building_split_pred draw_cell_pred draw_boundary_pred draw_cell_color

python scripts/test.py \
    logs/cpm17/config.py \
    logs/cpm17 \
    logs/cpm17/checkpoints/best.pth \
    --visual_pred \
    --draw_func draw_building_split_pred \
    --visual_interval 1
'''
