import os
import torch
import argparse
from densesam.utils import set_seed
from mmengine.config import Config
from tqdm import tqdm
import cv2
import numpy as np
from densesam.models.dense_sam import DenseSAMNet
import matplotlib.pyplot as plt
from mmdet.structures import DetDataSample
import torch.nn.functional as F

parser = argparse.ArgumentParser()

# base args
parser.add_argument('config_file', type=str)
parser.add_argument('result_save_dir', type=str)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('imgset_dir', type=str)
parser.add_argument('--compare', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--visual_interval', type=int, default=1)

args = parser.parse_args()

def draw_boundary_pred(img_path, pred_info, pred_save_dir):
    '''
    Args:
        pred_mask: tensor, shape is (h,w), h=w=1024
    '''
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pred_inst_mask = pred_info['pred_inst_mask']
    pred_boundary = pred_info['pred_boundary']

    fig = plt.figure(figsize=(12,4))
    h,w = pred_boundary.shape
    ax = fig.add_subplot(131)
    ax.imshow(img)
    ax.set_title('image')

    ax = fig.add_subplot(132)
    show_color_gt = np.zeros((h,w,4))
    show_color_gt[:,:,3] = 1
    inst_nums = len(np.unique(pred_inst_mask)) - 1
    for i in range(inst_nums):
        color_mask = np.concatenate([np.random.random(3), [1]])
        show_color_gt[pred_inst_mask==i+1] = color_mask
    ax.imshow(show_color_gt)
    ax.set_title('color pred')

    ax = fig.add_subplot(133)
    ax.imshow(pred_boundary, cmap='gray')
    ax.set_title('pred boundary')

    plt.tight_layout()
    plt.savefig(f'{pred_save_dir}/{image_name}')
    plt.close()


def inference_data():
    model.eval()
    MAXSIZE = 100

    images_name = os.listdir(args.imgset_dir)
    images_name = sorted(images_name)
    large_imgs = []
    for i_batch,image_name in enumerate(tqdm(images_name, ncols=70)):
        if i_batch % visual_args['visual_interval'] == 0:
            imgpath = os.path.join(args.imgset_dir, image_name)
            temp_image = cv2.imread(imgpath)
            h,w,_ = temp_image.shape
            if h > MAXSIZE and w > MAXSIZE:
                large_imgs.append(imgpath)
                imagesize = cv2.resize(temp_image, (1024,1024))
                img = np.array(imagesize, np.float32).transpose(2, 0, 1)
                img = torch.Tensor(img)  # (3, h, w)
                datainfo = DetDataSample()
                datainfo.ori_shape = (h,w)
                sampled_batch = dict(inputs = [img], data_samples = [datainfo])

                with torch.no_grad():
                    outputs = model(sampled_batch)
                logits = outputs['logits_256']
                for idx,datainfo in enumerate(sampled_batch['data_samples']):
                    origin_size = datainfo.ori_shape
                    logits_origin = F.interpolate(
                        logits[idx].unsqueeze(0),
                        origin_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)    # (k,o_h,o_w)
                    pred_result = (
                        torch.sigmoid(logits_origin[0,:,:].detach().unsqueeze(0)), 
                        torch.sigmoid(logits_origin[1,:,:].detach().unsqueeze(0))
                    )
                    pred_inst_mask, pred_boundary_mask = model.postprocess(datainfo, pred_result)
                    
                    pred_info = dict(
                        pred_inst_mask = pred_inst_mask,
                        pred_boundary = pred_boundary_mask
                    )
                    pred_save_dir = visual_args['pred_save_dir']
                    draw_boundary_pred(imgpath, pred_info, pred_save_dir)

    print(f'total images: {len(images_name)}, inference images: {len(large_imgs)}')


if __name__ == "__main__":
    device = torch.device(args.device)
    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)

    # register model
    model = DenseSAMNet(
        sm_depth = cfg.semantic_module_depth,
        use_inner_feat = cfg.use_inner_feat,
        use_boundary_head = cfg.use_boundary_head,
        use_embed = False,
        sam_ckpt = cfg.sam_ckpt,
        sam_type = cfg.sam_type,
        device = device,
        inter_idx = cfg.inter_idx
    ).to(device)

    visual_args = {}
    pred_save_dir = args.result_save_dir
    os.makedirs(pred_save_dir, exist_ok = True)
    visual_args['pred_save_dir'] = pred_save_dir
    visual_args['visual_interval'] = args.visual_interval

    # val
    model.load_parameters(args.ckpt_path)
    
    inference_data()



'''
python scripts/inference_nometric.py \
    logs/cnseg/2024_09_16_09_13_30/config.py \
    logs/alnet/clusteredCell2JFSW_large_aug \
    logs/cnseg/2024_09_16_09_13_30/checkpoints/best.pth \
    datasets/CervicalDatasets/JFSW/images \
    --visual_interval 1
'''
