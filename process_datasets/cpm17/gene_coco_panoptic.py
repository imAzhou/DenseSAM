from ..twochannel2coco_panoptic import converter

data_root = 'datasets/MedicalDatasets/CPM17'
size = 512

for overlap in [0]:
    for mode in ['train', 'test']:
        save_root_dir = f'{data_root}/{mode}_p{size}'
        if overlap > 0:
            save_root_dir = f'{data_root}/{mode}_p{size}_o{overlap}'

        prefix = 'panoptic_'
        
        source_folder = f'{save_root_dir}/{prefix}seg_anns'
        images_json_file = f'{save_root_dir}/{prefix}anns.json'
        segmentations_folder = f'{save_root_dir}/{prefix}seg_anns_coco'
        predictions_json_file = f'{save_root_dir}/{prefix}anns_coco.json'
        converter(source_folder, images_json_file, segmentations_folder, predictions_json_file)
        