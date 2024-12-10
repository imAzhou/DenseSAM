custom_imports = dict(imports=[
    'densesam.transforms.data_augmented',
    'densesam.transforms.pack_inputs'
], allow_failed_imports=False)

data_root = '/x22201018/datasets/CervicalDatasets/CNSeg'
dataset_tag = 'cnseg'
train_data_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args=None),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=False,
        with_mask=True,
        with_seg=True,
        backend_args=None),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    dict(type='ALNetTransform', input_size=(1024,1024), mode='train'),
    # dict(type='LoadBoundaryAnn', scale=(1024, 1024)),
    dict(type='PackMyInputs')
]

val_data_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args=None),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=False,
        with_mask=True,
        with_seg=True,
        backend_args=None),
    dict(type='ALNetTransform', input_size=(1024,1024), mode='test'),
    # dict(type='LoadBoundaryAnn'),
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

# train_load_parts = ['PatchSeg/train']
# val_load_parts = ['PatchSeg/test']
train_load_parts = ['clusteredCell/train']
val_load_parts = ['clusteredCell/test']

train_bs = 8
val_bs = 1
