custom_imports = dict(imports=[
    'densesam.transforms.data_augmented',
    'densesam.transforms.pack_inputs'
], allow_failed_imports=False)

data_root = 'datasets/MedicalDatasets/CoNIC'
dataset_tag = 'conic'
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
    dict(type='LoadBoundaryAnn'),
    dict(type='MyRotate', prob=0.5, max_mag=90.0, level=10,),
    dict(type='PackMyInputs')
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
    dict(type='LoadBoundaryAnn'),
    dict(type='PackMyInputs')
]

dataset = dict(
    ann_file = 'panoptic_anns_coco.json',
    data_prefix = dict(
        img = 'img_dir/', seg = 'panoptic_seg_anns_coco/'),
    filter_cfg = dict(filter_empty_gt=True, min_size=1),
    backend_args = None,
    load_embed = False,
)

train_load_parts = ['train']
val_load_parts = ['test']

train_bs = 8
val_bs = 8
