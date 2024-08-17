data_root = '/x22201018/datasets/MedicalDatasets/MoNuSeg'
dataset_tag = 'monuseg'
data_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args=None),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
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
    pipeline = data_pipeline,
    load_embed = True,
    load_inst = True
)

train_load_parts = ['train_p256']
val_load_parts = ['test_p256']
# train_load_parts = ['train_p256_o64']
# val_load_parts = ['test_p256_o64']

train_bs = 4
val_bs = 4
