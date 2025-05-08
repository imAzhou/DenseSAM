
import cv2
import os
import numpy as np

def draw_grid_on_image(image_path, img_size, grid_size, output_path):
    """
    在图片上绘制指定网格大小的黑色线。

    Args:
        image_path (str): 输入图片路径。
        grid_size (int): 网格大小（每个格子的宽度和高度）。
        output_path (str): 输出图片路径。
    """
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_size,img_size))
    height, width, _ = image.shape

    # 绘制水平线
    for y in range(0, height, grid_size):
        cv2.line(image, (0, y), (width, y), (0, 0, 0), 1)

    # 绘制垂直线
    for x in range(0, width, grid_size):
        cv2.line(image, (x, 0), (x, height), (0, 0, 0), 1)

    # 保存结果
    cv2.imwrite(output_path, image)


if __name__ == '__main__':

    img_dir = 'datasets/MedicalDatasets/CPM17/test_p512/img_dir'
    img_name = 'image_00_3'
    img_path = f'{img_dir}/{img_name}.png'
    save_dir = f'visual_results/patchsize_in_image'
    os.makedirs(save_dir, exist_ok=True)
    
    img_size = 1024
    patch_size = 16
    save_path = f'{save_dir}/{img_name}.png'

    draw_grid_on_image(img_path, img_size, patch_size, save_path)