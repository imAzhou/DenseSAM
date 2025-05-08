custom_imports = dict(imports=[
    'densesam.transforms.data_augmented',
    'densesam.transforms.pack_inputs'
], allow_failed_imports=False)

data_root = 'datasets/MedicalDatasets/CPM17'
dataset_tag = 'cpm17'
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
    # dict(type='Albu', 
    #     transforms=[
    #         dict(type='Blur', blur_limit=10, p=0.2),
    #         dict(type='GaussNoise', var_limit=50, p=0.25),
    #         dict(type='ColorJitter', brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05, p=0.2)],
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
    #     keymap={
    #         'img': 'image',
    #         'gt_bboxes': 'bboxes'
    #         }),
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

train_load_parts = ['train_p512']
val_load_parts = ['test_p512']

train_bs = 8
val_bs = 8
