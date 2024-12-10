data_root = '/x22201018/datasets/LIP'
dataset_tag = 'lip'

data_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args=None),
    # dict(type='Pad', pad_to_square=True),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        backend_args=None),
    dict(type='Pad', pad_to_square=True, pad_val=dict(img=0, seg=0)),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackDetInputs')
]

train_data_pipeline = data_pipeline
val_data_pipeline = data_pipeline

dataset = dict(
    ann_file = 'panoptic_anns_coco.json',
    data_prefix = dict(
        img = 'img_dir/', seg = 'panoptic_seg_anns_coco/'),
    filter_cfg = dict(filter_empty_gt=True, min_size=1),
    backend_args = None,
    load_embed = True
)

train_load_parts = ['train']
val_load_parts = ['val']

train_bs = 16
val_bs = 16
