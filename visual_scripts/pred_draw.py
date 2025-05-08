from densesam.utils import show_multi_mask,show_mask
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import random
from mmdet.structures import DetDataSample

def draw_boundary_pred(datainfo:DetDataSample, gt_info, pred_info,pred_save_dir):
    '''
    Args:
        pred_mask: tensor, shape is (h,w), h=w=1024
    '''
    img_path = datainfo.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    gt_mask = datainfo.gt_sem_seg.sem_seg[0]
    gt_inst_mask = gt_info['gt_inst_mask']
    gt_boundary = gt_info['gt_boundary']
    
    pred_mask = pred_info['pred_mask']
    pred_inst_mask = pred_info['pred_inst_mask']
    # pred_boundary = pred_info['pred_boundary'].detach().cpu()
    pred_boundary = pred_info['pred_boundary']

    rgbs = [[255,255,255],[47, 243, 15]]

    fig = plt.figure(figsize=(12,9))
    h,w = gt_mask.shape
    ax = fig.add_subplot(231)
    ax.imshow(img)
    show_multi_mask(gt_mask.cpu(), ax, palette = rgbs)
    ax.set_title('gt mask')
    ax = fig.add_subplot(232)
    show_color_gt = np.zeros((h,w,4))
    show_color_gt[:,:,3] = 1
    inst_nums = len(np.unique(gt_inst_mask)) - 1
    for i in range(inst_nums):
        color_mask = np.concatenate([np.random.random(3), [1]])
        show_color_gt[gt_inst_mask==i+1] = color_mask
    ax.imshow(show_color_gt)
    ax.set_axis_off()
    ax.set_title('color gt')
    ax = fig.add_subplot(233)
    ax.imshow(gt_boundary, cmap='gray')
    ax.set_title('gt boundary')
    
    ax = fig.add_subplot(234)
    ax.imshow(img)
    show_multi_mask(pred_mask, ax, palette = rgbs)
    ax.set_title('pred mask')
    ax = fig.add_subplot(235)
    ax.imshow(pred_mask, cmap='gray')
    # show_color_gt = np.zeros((h,w,4))
    # show_color_gt[:,:,3] = 1
    # inst_nums = len(np.unique(pred_inst_mask)) - 1
    # for i in range(inst_nums):
    #     color_mask = np.concatenate([np.random.random(3), [1]])
    #     show_color_gt[pred_inst_mask==i+1] = color_mask
    # ax.imshow(show_color_gt)
    # ax.set_title('color pred')
    ax = fig.add_subplot(236)
    ax.imshow(pred_boundary, cmap='gray')
    ax.set_title('pred boundary')

    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()

def draw_cell_pred(datainfo:DetDataSample, gt_info, pred_info,pred_save_dir):
    '''
    Args:
        pred_mask: tensor, shape is (h,w)
    '''
    img_path = datainfo.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt_inst_mask = gt_info['gt_inst_mask']
    pred_inst_mask = pred_info['pred_inst_mask']

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121)
    inst_nums = len(np.unique(gt_inst_mask)) - 1
    copy_img = copy.deepcopy(img)
    for i in range(inst_nums):
        color_mask = [random.randint(0, 255) for _ in range(3)]
        copy_img[gt_inst_mask==i+1] = color_mask
    ax.imshow(copy_img)
    ax.set_axis_off()
    ax.set_title('color gt')

    ax = fig.add_subplot(122)
    inst_nums = len(np.unique(pred_inst_mask)) - 1
    copy_img = copy.deepcopy(img)
    for i in range(inst_nums):
        color_mask = [random.randint(0, 255) for _ in range(3)]
        copy_img[pred_inst_mask==i+1] = color_mask
    ax.imshow(copy_img)
    ax.set_axis_off()
    ax.set_title('color pred')
    
    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()

def draw_cell_color(datainfo:DetDataSample, gt_info, pred_info,pred_save_dir):
    '''
    Args:
        pred_mask: tensor, shape is (h,w)
    '''
    img_path = datainfo.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt_inst_mask = gt_info['gt_inst_mask']
    pred_inst_mask = pred_info['pred_inst_mask']
    h,w = gt_inst_mask.shape

    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(131)
    ax.imshow(img)
    ax.set_title('input image')

    ax = fig.add_subplot(132)
    show_color_gt = np.zeros((h,w,4))
    show_color_gt[:,:,3] = 1
    inst_nums = len(np.unique(gt_inst_mask)) - 1
    for i in range(inst_nums):
        color_mask = np.concatenate([np.random.random(3), [1]])
        show_color_gt[gt_inst_mask==i+1] = color_mask
    ax.imshow(show_color_gt)
    ax.set_axis_off()
    ax.set_title('color gt')

    ax = fig.add_subplot(133)
    show_color_gt = np.zeros((h,w,4))
    show_color_gt[:,:,3] = 1
    inst_nums = len(np.unique(pred_inst_mask)) - 1
    for i in range(inst_nums):
        color_mask = np.concatenate([np.random.random(3), [1]])
        show_color_gt[pred_inst_mask==i+1] = color_mask
    ax.imshow(show_color_gt)
    ax.set_title('color pred')
    
    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()


def draw_building_pred(datainfo:DetDataSample, pred_info,pred_save_dir):
    '''
    Args:
        pred_mask: tensor, shape is (h,w)
    '''
    img_path = datainfo.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb = [47, 243, 15]

    gt_mask = datainfo.gt_sem_seg.sem_seg
    pred_mask = pred_info['pred_mask']

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121)
    ax.imshow(img)
    show_mask(gt_mask.cpu(), ax, rgb=rgb)
    ax.set_title('gt mask')

    ax = fig.add_subplot(122)
    ax.imshow(img)
    show_mask(pred_mask.cpu(), ax, rgb=rgb)
    ax.set_title('pred mask')

    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()


def draw_building_split_pred(datainfo:DetDataSample, pred_info,pred_save_dir):
    '''
    Args:
        pred_mask: tensor, shape is (h,w)
    '''
    img_path = datainfo.img_path
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb = [47, 243, 15]

    gt_mask = datainfo.gt_sem_seg.sem_seg
    pred_mask = pred_info['pred_mask']

    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(131)
    ax.imshow(img)
    ax.set_title('image')
    ax.set_axis_off()

    ax = fig.add_subplot(132)
    ax.imshow(gt_mask[0].cpu().numpy(), cmap='gray')
    ax.set_title('gt mask')
    ax.set_axis_off()

    ax = fig.add_subplot(133)
    ax.imshow(pred_mask.cpu().numpy(), cmap='gray')
    ax.set_title('pred mask')
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()

