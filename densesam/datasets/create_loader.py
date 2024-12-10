from torch.utils.data import DataLoader, ConcatDataset
from mmengine.dataset.sampler import DefaultSampler
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.config import Config
from .cell_binary import CellBinaryDataset
from .building_binary import BuildingBinaryDataset

support_datasets = ['whu', 'inria', 'mass', 'lip', 'pannuke_binary', 'monuseg', 'cpm17', 'conic', 'cnseg']

def gene_loader_trainval(
        *,
        dataset_config: Config,
        seed
):
    dataset_tag = dataset_config.get('dataset_tag', None)
    
    assert dataset_tag in support_datasets, \
        f'the dataset_tag must be in {support_datasets}'
    
    dataloader_config = dict(num_workers = 16, seed = seed)
    data_root = dataset_config['data_root']
    
    dataset_config['load_parts'] = dataset_config['train_load_parts']
    dataloader_config['batch_size'] = dataset_config['train_bs']
    dataset_config['dataset']['pipeline'] = dataset_config['train_data_pipeline']
    train_dataloader,metainfo = create_dataset_loader(dataset_config, dataloader_config, dataset_tag)

    dataset_config['load_parts'] = dataset_config['val_load_parts']
    dataloader_config['batch_size'] = dataset_config['val_bs']
    dataset_config['dataset']['pipeline'] = dataset_config['val_data_pipeline']
    val_dataloader,metainfo = create_dataset_loader(dataset_config, dataloader_config, dataset_tag)

    part = dataset_config['val_load_parts'][0]
    root_path = f'{data_root}/{part}'

    panoptic_ann_file = f'{root_path}/panoptic_anns_coco.json'
    detection_ann_file = f'{root_path}/detection_anns_coco.json'
    seg_prefix = f'{root_path}/panoptic_seg_anns_coco'

    if dataset_tag == 'pannuke_binary':
        panoptic_ann_file = f'{root_path}/panoptic_binary_anns_coco.json'
        detection_ann_file = f'{root_path}/detection_binary_anns_coco.json'
        seg_prefix = f'{root_path}/panoptic_binary_seg_anns_coco'
    
    restinfo = dict(
        panoptic_ann_file = panoptic_ann_file,
        detection_ann_file = detection_ann_file,
        seg_prefix = seg_prefix,
        root_path = root_path
    )

    return train_dataloader, val_dataloader, metainfo, restinfo



def gene_loader_eval(
        *,
        dataset_config: Config,
        seed
):
    dataset_tag = dataset_config.get('dataset_tag', None)
    assert dataset_tag in support_datasets, \
        f'the dataset_tag must be in {support_datasets}'
    
    dataloader_config = dict(num_workers = 16, seed = seed)
    data_root = dataset_config['data_root']

    dataset_config['load_parts'] = dataset_config['val_load_parts']
    dataloader_config['batch_size'] = dataset_config['val_bs']
    dataset_config['dataset']['pipeline'] = dataset_config['val_data_pipeline']
    dataloader,metainfo = create_dataset_loader(dataset_config, dataloader_config, dataset_tag)

    part = dataset_config['val_load_parts'][0]
    root_path = f'{data_root}/{part}'

    panoptic_ann_file = f'{root_path}/panoptic_anns_coco.json'
    detection_ann_file = f'{root_path}/detection_anns_coco.json'
    seg_prefix = f'{root_path}/panoptic_seg_anns_coco'

    if dataset_tag == 'pannuke_binary':
        panoptic_ann_file = f'{root_path}/panoptic_binary_anns_coco.json'
        detection_ann_file = f'{root_path}/detection_binary_anns_coco.json'
        seg_prefix = f'{root_path}/panoptic_binary_seg_anns_coco'
    
    restinfo = dict(
        panoptic_ann_file = panoptic_ann_file,
        detection_ann_file = detection_ann_file,
        seg_prefix = seg_prefix,
        root_path = root_path
    )
    return dataloader,metainfo,restinfo
    

def create_dataset_loader(
        dataset_config: dict, dataloader_config: dict, dataset_tag: str):

    init_default_scope('mmdet')

    load_parts = dataset_config['load_parts']
    data_root = dataset_config['data_root']

    all_datasets = []
    for part in load_parts:
        d_cfg = dataset_config.dataset
        d_cfg.data_root = f'{data_root}/{part}'
        
        if dataset_tag in ['whu', 'inria', 'mass', 'lip']:
            dataset = BuildingBinaryDataset(**d_cfg)
        
        if dataset_tag in ['pannuke_binary', 'monuseg', 'cpm17', 'conic', 'cnseg']:
            dataset = CellBinaryDataset(**d_cfg)
        
        all_datasets.append(dataset)

    combined_dataset = None
    if len(all_datasets) > 1:
        combined_dataset = ConcatDataset(all_datasets)
        combined_dataset.METAINFO = all_datasets[0].METAINFO
    else:
        combined_dataset = all_datasets[0]
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size = dataloader_config['batch_size'],
        num_workers = dataloader_config['num_workers'],
        drop_last = False,
        persistent_workers = True,
        sampler = DefaultSampler(dataset=combined_dataset, shuffle=True, seed=dataloader_config['seed']),
        collate_fn = pseudo_collate
    )

    metainfo = combined_dataset.METAINFO

    return dataloader,metainfo
