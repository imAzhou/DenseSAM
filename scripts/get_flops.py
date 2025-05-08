import torch
from mmengine.config import Config
from mmengine.analysis import get_model_complexity_info
from densesam.models.dense_sam import DenseSAMNet

def get_flops():
    device = torch.device('cpu')
    config_file = 'logs/cpm17/config.py'
    cfg = Config.fromfile(config_file)

    # model = DenseSAMNet(
    #         sm_depth = cfg.semantic_module_depth,
    #         use_inner_feat = cfg.use_inner_feat,
    #         use_boundary_head = cfg.use_boundary_head,
    #         use_embed = False,
    #         sam_ckpt = cfg.sam_ckpt,
    #         sam_type = cfg.sam_type,
    #         device = device,
    #         inter_idx = cfg.inter_idx
    #     )
    # img_embed = torch.load('datasets/MedicalDatasets/CPM17/train_p512/img_tensor/image_00_0.pt')
    # inter_feat = torch.load('datasets/MedicalDatasets/CPM17/train_p512/img_tensor/image_00_0_inner_1.pt')
    # sampled_batch = dict(img_embed=[img_embed], inter_feat=[inter_feat])
    
    model = DenseSAMNet(
            sm_depth = 0,
            use_inner_feat = False,
            use_boundary_head = False,
            use_embed = True,
            sam_ckpt = cfg.sam_ckpt,
            sam_type = cfg.sam_type,
            device = device,
            inter_idx = cfg.inter_idx
        )
    img_embed = torch.load('datasets/MedicalDatasets/CPM17/train_p512/img_tensor/image_00_0.pt')
    inter_feat = torch.load('datasets/MedicalDatasets/CPM17/train_p512/img_tensor/image_00_0_inner_1.pt')
    sampled_batch = dict(img_embed=[img_embed], inter_feat=[inter_feat])
    
    analysis_results = get_model_complexity_info(
        model,
        None,
        inputs=sampled_batch,
    )

    flops = analysis_results['flops_str']
    params = analysis_results['params_str']
    activations = analysis_results['activations_str']
    out_table = analysis_results['out_table']
    out_arch = analysis_results['out_arch']
    print(out_arch)
    print(out_table)
    split_line = '=' * 30
    print(f'{split_line}\n'
            f'Flops: {flops}\nParams: {params}\n'
            f'Activation: {activations}\n{split_line}')


if __name__ == '__main__':
    get_flops()
