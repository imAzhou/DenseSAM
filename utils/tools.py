import random
import numpy as np
import torch
import numpy as np
import cv2
import copy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    one_M = 1e6
    return {
        'Total': f'{(total_num/one_M):.4f}M',
        'Trainable': f'{(trainable_num/one_M):.4f}M',
    }

def one_hot_encoder(n_classes, input_tensor):
    '''
    Args:
        n_classes (int): number of classess
        input_tensor (Tensor): shape is (bs, h, w), 
            the pixel value belong to [0, n_classes-1]
    Return:
        output_tensor (Tensor): shape is (bs, n_classes, h, w), 
            the pixel value belong to [0, 1]
    '''
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def gene_points(image_mask, pos_sample_num, neg_sample_num):
    '''
    Args:
        image_mask(Tensor): The image masks. Expects an
            image in HW uint8 format, with pixel values in [0, 1].
    return:
        input_point = np.array([[x, y], ..., [x, y]])
        input_label = np.array([1, ..., 0])
    '''
    input_point = []
    input_label = []
    # 获取前景点的所有坐标
    # positive_coords.shape: (total_positive_num, 2)
    positive_coords = np.nonzero(image_mask)
    if positive_coords.shape[0] > 0:
        # 有可能取的正样本点数量超过所有的前景像素点，replace=True : 可以重复选点
        random_pos_index = np.random.choice(np.arange(positive_coords.shape[0]), size=pos_sample_num,replace=True)
    else:
        random_pos_index = []
    # 获取背景点的所有坐标
    # negative_coords: tuple(np.array(Y),np.array(X))
    negative_coords = np.where(image_mask == 0)
    random_neg_index = np.random.choice(np.arange(len(negative_coords[0])), size=neg_sample_num,replace=True)
    
    # 返回坐标格式为 (x,y), image_mask的格式为 H*W，故其第一个值为 y，第二个值为x
    for idx in random_pos_index:
        input_point.append([positive_coords[idx][1].item(),positive_coords[idx][0].item()])
        input_label.append(1)
    
    for idx in random_neg_index:
        input_point.append([negative_coords[1][idx],negative_coords[0][idx]])
        input_label.append(0)

    return np.array(input_point),np.array(input_label)


def gene_bbox_for_mask(image_mask_np):
    '''
    Args:
        image_mask_np(numpy): The image masks. Expects an
            image in HW uint8 format, with pixel values in [0, 1].
    return:
        [[x1,y1,x2,y2], ..., [x1,y1,x2,y2]]
    '''
    # 寻找连通域
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mask_np, connectivity=8)
    trans_box = lambda x1,y1,w,h: [x1,y1, x1 + w, y1 + h]
    # stats[0] 是背景框
    all_boxes = [trans_box(x1,y1,w,h) for x1,y1,w,h,_ in stats[1:]]
    return all_boxes


def get_prompt(prompt_type, binary_mask, boxes_candidate, device, coord_ratio=1):
    '''
    Notices:
        - binary_mask and boxes_candidate can only exist one of them.
        - if boxes_candidate is None and prompt type include box, the boxes_candidate will be
          generate from binary_mask.
        - the return values of boxes and points, there dim 0 means different
    
    Args:
        - prompt_type(string): prompt type, belong to 
            ['max_bbox', 'random_bbox', 'all_bboxes', 'random_point', 'max_bbox_center_point']
        - binary_mask(np.array): The image masks. Expects an
            image in HW uint8 format, with pixel values in [0, 1].
        - boxes_candidate(np.array): The candidate boxes. Expects value format is 
            [[x1,y1,x2,y2], ..., [x1,y1,x2,y2]].
        - device: tensor to device.
        - coord_ratio(int): box or point coords will be scaled to coord_ratio
    
    Returns:
        (boxes and point which meet requirements of SAM prompt encoder)
        - boxes(tensor): shape is (k, 1, 4), k is the num of boxes, 4 is (x1, y1, x2, y2)
        - point(tuple): (coords_torch, labels_torch), coords_torch shape is (bs, k, 2), 2 is (x,y)
            labels_torch shape is (bs, k), k is the num of points.
        - sampled_idx(tensor): the idx of sampled boxes or point.
    '''
    allowed_types = ['max_bbox', 'random_bbox', 'all_bboxes', 'random_point', 'max_bbox_center_point', 'max_bbox_with_point']
    assert prompt_type in allowed_types, \
            f'the prompt_type must be in {allowed_types}'
    assert binary_mask is not None or boxes_candidate is not None, \
            f'binary_mask and boxes_candidate can only exist one of them'
    
    
    boxes, point, sampled_idx = None, None, None

    if binary_mask is not None and np.sum(binary_mask) == 0:
        return boxes, point, sampled_idx
    if boxes_candidate is not None and len(boxes_candidate) == 0:
        return boxes, point, sampled_idx

    if prompt_type == 'random_point':
        input_points,_ = gene_points(binary_mask, 1, 0)
        random_point = input_points[0] * coord_ratio
        coords_torch = torch.as_tensor(np.array([random_point]), dtype=torch.float, device=device)
        labels_torch = torch.ones(len(coords_torch), dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        point = (coords_torch, labels_torch)
        return boxes, point, None
    
    if boxes_candidate is None:
        boxes_candidate = gene_bbox_for_mask(binary_mask)
        boxes_candidate = np.array(boxes_candidate)
    else:
        boxes_candidate = copy.deepcopy(boxes_candidate)
    
    boxes_candidate *= coord_ratio

    if prompt_type == 'random_bbox':
        random_bbox_idx = np.random.choice(range(len(boxes_candidate)), 1)[0]
        box_np = np.array(boxes_candidate[random_bbox_idx]) # [x1,y1,x2,y2]
        box_torch = torch.as_tensor(box_np[None, :], dtype=torch.float, device=device)  #[[x1,y1,x2,y2]]
        boxes = box_torch[None, :]  #[[[x1,y1,x2,y2]]]
        return boxes, point, [random_bbox_idx]
    if prompt_type == 'all_bboxes':
        box_np = np.array(boxes_candidate)
        boxes = torch.as_tensor(box_np, dtype=torch.float, device=device)
        boxes = boxes.unsqueeze(1)  # (k_box_nums, 1, 4)
        return boxes, point, np.arange(len(boxes_candidate))
   
    
    boxes_area = np.array([(x2-x1) * (y2-y1) for x1,y1,x2,y2 in boxes_candidate])
    max_idx = np.argmax(boxes_area)
    if prompt_type == 'max_bbox':
        box_np = np.array(boxes_candidate[max_idx]) # [x1,y1,x2,y2]
        box_torch = torch.as_tensor(box_np[None, :], dtype=torch.float, device=device)  #[[x1,y1,x2,y2]]
        boxes = box_torch[None, :]  #[[[x1,y1,x2,y2]]]
        return boxes, point, [max_idx]
    if prompt_type == 'max_bbox_center_point':
        x1,y1,x2,y2 = boxes_candidate[max_idx]
        centroid_x,centroid_y = int(x1+(x2-x1)/2),int(y1+(y2-y1)/2)
        coords_torch = torch.as_tensor(np.array([[centroid_x,centroid_y]]), dtype=torch.float, device=device)
        labels_torch = torch.ones(len(coords_torch), dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        point = (coords_torch, labels_torch)
        return boxes, point, [max_idx]
    if prompt_type == 'max_bbox_with_point':
        box_np = np.array(boxes_candidate[max_idx]) # [x1,y1,x2,y2]
        box_torch = torch.as_tensor(box_np[None, :], dtype=torch.float, device=device)  #[[x1,y1,x2,y2]]
        boxes = box_torch[None, :]  #[[[x1,y1,x2,y2]]]
    
        x1,y1,x2,y2 = boxes_candidate[max_idx]
        centroid_x,centroid_y = int(x1+(x2-x1)/2),int(y1+(y2-y1)/2)
        coords_torch = torch.as_tensor(np.array([[centroid_x,centroid_y]]), dtype=torch.float, device=device)
        labels_torch = torch.ones(len(coords_torch), dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        point = (coords_torch, labels_torch)
        return boxes, point, [max_idx]

