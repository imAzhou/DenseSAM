import os
import cv2
import json
import numpy as np
from tqdm import tqdm

# random.seed(2)
root = '/x22201018/datasets/MedicalDatasets/PanNuke'
parts = ['Part1', 'Part2', 'Part3']

colors = {
    0: (255, 0, 0),   # 癌变细胞 Neoplastic cells - 红色
    1: (0, 255, 0),   # 炎症细胞 Inflammatory - 绿色
    2: (0, 0, 255),   # 结缔组织/软组织细胞 Connective/Soft tissue cells - 蓝色
    3: (255, 255, 0),  # 死亡细胞 Dead Cells - 黄色
    4: (255, 128, 0),  # 上皮细胞 Epithelial - 橙色
    # 5: (0, 0, 0)      # 背景 Background - 黑色
}
palette=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 128, 0], [255, 255, 255]]

def gene_mask_png():
    for idx, part in enumerate(parts):
        masks_dict_path = os.path.join(root, part, 'Masks', 'masks.json')
        image_path = os.path.join(root, part, 'Images', 'images.npy')
        mask_path = os.path.join(root, part, 'Masks', 'masks.npy')
        images = np.load(image_path)    # shape: (total_img_nums, h, w, c)
        masks = np.load(mask_path)      # shape: (total_img_nums, h, w, cls_num)

        print(f'images:{images.shape}, masks:{masks.shape}')
        
        ann_dir = f'{root}/{part}/ann_dir'
        os.makedirs(ann_dir, exist_ok=True)
        img_ann_dir = f'{root}/{part}/Masks/img_ann_dir'
        os.makedirs(img_ann_dir, exist_ok=True)
        img_dir = f'{root}/{part}/img_dir'

        with open(masks_dict_path,'r',encoding='utf-8') as f :
            masks_dict = json.load(f)
            for filename,mask_idx in tqdm(masks_dict.items(), ncols=70):
                img = cv2.imread(f'{img_dir}/{filename}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask_png = np.zeros(img.shape[:2])  # 创建一个全零的颜色掩码
                mask_ann = masks[mask_idx]
                for i in range(len(palette)):
                    mask_png[mask_ann[...,i] > 0] = i+1
                
                cv2.imwrite(f'{ann_dir}/{filename}', mask_png)

                # fig = plt.figure(figsize=(6,6))
                # ax = fig.add_subplot(111)
                # ax.imshow(img)
                # vis_mask_png = copy.deepcopy(mask_png)
                # vis_mask_png -= 1
                # vis_mask_png[vis_mask_png==-1] = 255
                # show_multi_mask(vis_mask_png, ax, palette)
                # ax.set_title('gt mask')
                # plt.tight_layout()
                # plt.savefig(f'{img_ann_dir}/{filename}')
                # plt.close()

def rename_file():
    for idx, part in enumerate(parts):
        img_dir = f'{root}/{part}/img_dir'
        ann_dir = f'{root}/{part}/ann_dir'
        all_img_filename = os.listdir(img_dir)
        for filename in tqdm(all_img_filename,ncols=70):
            old_img_file = os.path.join(img_dir, filename)
            old_ann_file = os.path.join(ann_dir, filename)
            new_filename = f'{part}_{filename}'
            new_img_file = os.path.join(img_dir, new_filename)
            new_ann_file = os.path.join(ann_dir, new_filename)
            # 重命名文件
            os.rename(old_img_file, new_img_file)
            os.rename(old_ann_file, new_ann_file)


def get_boxes_from_mask(mask_with_instanceid):
    all_instance_id = np.unique(mask_with_instanceid)
    all_boxes = []
    for instance_id in all_instance_id:
        if instance_id > 0:
            instance_mask = (mask_with_instanceid == instance_id).astype(np.uint8)
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 确保至少找到一个轮廓
            if len(contours) > 0:
                # 获取第一个轮廓
                contour = contours[0]
                # 计算包围矩形的坐标
                x1, y1, w, h = cv2.boundingRect(contour)
                # print(f"包围矩形坐标: x={x1}, y={y1}, w={w}, h={h}")
                x2 = x1 + w
                y2 = y1 + h
                all_boxes.append([x1,y1,x2,y2])
    return all_boxes

def gene_box_json():
    for idx, part in enumerate(parts):
        masks_dict_path = os.path.join(root, part, 'Masks', 'masks.json')
        mask_path = os.path.join(root, part, 'Masks', 'masks.npy')
        masks = np.load(mask_path)      # shape: (total_img_nums, h, w, cls_num)

        ann_boxes_dir = f'{root}/{part}/ann_boxes_dir'
        os.makedirs(ann_boxes_dir, exist_ok=True)
        img_ann_dir = f'{root}/{part}/Masks/img_ann_dir_with_boxes'
        os.makedirs(img_ann_dir, exist_ok=True)
        img_dir = f'{root}/{part}/img_dir'

        with open(masks_dict_path,'r',encoding='utf-8') as f :
            masks_dict = json.load(f)
            for filename,mask_idx in tqdm(masks_dict.items(), ncols=70):
                boxes_json = {}
                img = cv2.imread(f'{img_dir}/{filename}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask_png = np.zeros(img.shape[:2])  # 创建一个全零的颜色掩码
                mask_ann = masks[mask_idx]  # shape: (h, w, cls_num)
                for i in range(len(palette)):
                    mask_png[mask_ann[...,i] > 0] = i+1
                    # 不考虑背景类的 box
                    if i != len(palette)-1:
                        all_instance_boxes = get_boxes_from_mask(mask_ann[...,i])
                        if len(all_instance_boxes) > 0:
                            boxes_json[f'{i}'] = all_instance_boxes
                json_name = filename.split('.')[0] + '.json'
                json_save_path = f'{ann_boxes_dir}/{json_name}'
                with open(json_save_path, 'w', encoding='utf-8') as mask_f:
                    json.dump(boxes_json, mask_f)

                # fig = plt.figure(figsize=(6,6))
                # ax = fig.add_subplot(111)
                # ax.imshow(img)
                # vis_mask_png = copy.deepcopy(mask_png)
                # vis_mask_png -= 1
                # vis_mask_png[vis_mask_png==-1] = 255
                # show_multi_mask(vis_mask_png, ax, palette)
                # for cls_id,boxes in boxes_json.items():
                #     cls_id = int(cls_id)
                #     for box in boxes:
                #         edgecolor = np.array([palette[cls_id][0]/255, palette[cls_id][1]/255, palette[cls_id][2]/255, 1])
                #         show_box(box, ax, edgecolor=edgecolor)
                # ax.set_title('gt mask')
                # plt.tight_layout()
                # plt.savefig(f'{img_ann_dir}/{filename}')
                # plt.close()


if __name__ == '__main__':
    # gene_mask_png()
    gene_box_json()