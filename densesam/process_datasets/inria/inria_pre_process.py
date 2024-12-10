import shutil
import os
from tqdm import tqdm

city_names = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
val_tifs = []
for cityname in city_names:
    for i in range(5):
        val_tifs.append(f'{cityname}{i+1}.tif')

root_dir = '/x22201018/datasets/RemoteSensingDatasets/InriaBuildingDataset'
all_img_dir = f'{root_dir}/train/images'
all_gt_dir = f'{root_dir}/train/gt'
save_train_img_dir = f'{root_dir}/train_images'
save_val_img_dir = f'{root_dir}/val_images'
save_train_gt_dir = f'{root_dir}/train_masks'
save_val_gt_dir = f'{root_dir}/val_masks'

for filename in tqdm(os.listdir(all_img_dir)):
    img_output_folder = save_val_img_dir if filename in val_tifs else save_train_img_dir
    gt_output_folder = save_val_gt_dir if filename in val_tifs else save_train_gt_dir
    # 确保输出文件夹存在
    os.makedirs(img_output_folder, exist_ok=True)
    os.makedirs(gt_output_folder, exist_ok=True)

    origin_img, origin_gt = f'{all_img_dir}/{filename}', f'{all_gt_dir}/{filename}'
    target_img, target_gt = f'{img_output_folder}/{filename}', f'{gt_output_folder}/{filename}'
    shutil.copyfile(origin_img, target_img)
    shutil.copyfile(origin_gt, target_gt)