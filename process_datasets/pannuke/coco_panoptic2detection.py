from process_datasets.panoptic2detection import convert_panoptic_to_detection_coco_format

data_root = '/x22201018/datasets/MedicalDatasets/PanNuke'
'''
python process_datasets/pannuke/panoptic2detection.py \
  --input_json_file /x22201018/datasets/MedicalDatasets/PanNuke/Part3/panoptic_binary_anns_coco.json \
  --segmentations_folder /x22201018/datasets/MedicalDatasets/PanNuke/Part3/panoptic_binary_seg_anns_coco \
  --output_json_file /x22201018/datasets/MedicalDatasets/PanNuke/Part3/detection_binary_anns_coco.json \
  --things_only
'''

for type in ['binary', 'origin']:
    for part in ['Part1', 'Part2', 'Part3']:
        prefix = 'panoptic_'
        output_prefix = 'detection_'
        if type == 'binary':
            prefix += 'binary_'
            output_prefix += 'binary_'

        input_json_file = f'{data_root}/{part}/{prefix}anns_coco.json'
        segmentations_folder = f'{data_root}/{part}/{prefix}seg_anns_coco'
        output_json_file = f'{data_root}/{part}/{output_prefix}anns_coco.json'
        things_only = True
        convert_panoptic_to_detection_coco_format(
            input_json_file, segmentations_folder, output_json_file, things_only)
        