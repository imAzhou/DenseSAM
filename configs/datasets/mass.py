custom_imports = dict(imports=[
    'densesam.transforms.data_augmented',
], allow_failed_imports=False)

data_root = 'datasets/RemoteSensingDatasets/MassachusettsBuilding'
dataset_tag = 'mass'
train_data_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args=None),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        backend_args=None),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='MyRotate', prob=0.5, max_mag=90.0, level=10,),
    dict(type='PackDetInputs')
]

val_data_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args=None),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        backend_args=None),
    dict(type='PackDetInputs')
]

dataset = dict(
    ann_file = 'panoptic_anns_coco.json',
    data_prefix = dict(
        img = 'img_dir/', seg = 'panoptic_seg_anns_coco/'),
    filter_cfg = dict(filter_empty_gt=True, min_size=1),
    backend_args = None,
    load_embed = False
)

train_load_parts = ['train_p500']
val_load_parts = ['test_p500']

train_bs = 8
val_bs = 8
