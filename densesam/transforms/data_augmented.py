import numpy as np
import albumentations as A
import cv2
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


def colortransforms(image):
    hue_shift = np.random.randint(10,50)
    sat_shift = np.random.randint(10,50)
    val_shift = np.random.randint(10,50)

    transform = A.Compose([
        A.HueSaturationValue(hue_shift_limit=hue_shift, sat_shift_limit=sat_shift, val_shift_limit=val_shift, always_apply=False, p=0.5),
        A.RGBShift(r_shift_limit=hue_shift, g_shift_limit=sat_shift, b_shift_limit=val_shift, always_apply=False, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
        A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        A.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
        A.RandomShadow (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=0.5),
        A.RandomSnow (snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=0.5),
        A.RandomSunFlare (flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False, p=0.5),
        A.RandomToneCurve (scale=0.1, always_apply=False, p=0.5),
        A.Solarize (threshold=128, always_apply=False, p=0.5),
        A.Blur (blur_limit=7, always_apply=False, p=0.5),
        A.Downscale (scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=0.5),
        A.Equalize (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5),
        A.FancyPCA (alpha=0.1, always_apply=False, p=0.5),
        A.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
        A.GridDropout (ratio=0.5, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None, shift_x=0, shift_y=0, random_offset=False, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
        A.RandomGridShuffle (grid=(3, 3), always_apply=False, p=0.5),
    ])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,ring,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0,0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(0, 0,0,))
        ring = cv2.warpPerspective(ring, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(0, 0,0,))
    return image, mask,ring

def randomHorizontalFlip(image, mask,ring, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        ring = cv2.flip(ring, 1)
    return image, mask,ring

def randomVerticleFlip(image, mask, ring,u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        ring = cv2.flip(ring, 0)
    return image, mask,ring

def randomRotate90(image, mask,ring, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
        ring=np.rot90(ring)
    return image, mask,ring


def transform_datas(datas, input_size, mode):
    '''
    Args:
        datas: (image, mask, boundary), np.array
            image: (h, w, bgr),
            mask: (h,w),
            boundary: (h,w),
    '''
    image, mask, boundary = datas
    image = cv2.resize(image.astype(np.uint8), input_size)
    
    if mode == 'train':
        mask = cv2.resize(mask.astype(np.uint8), input_size)
        boundary = cv2.resize(boundary.astype(np.uint8), input_size)
        image = colortransforms(image)
        image = randomHueSaturationValue(image,
                                    hue_shift_limit=(-30, 30),
                                    sat_shift_limit=(-5, 5),
                                    val_shift_limit=(-15, 15))
        image, mask ,boundary = randomShiftScaleRotate(image, mask,boundary,
                                        shift_limit=(-0.1, 0.1),
                                        scale_limit=(-0.1, 0.1),
                                        aspect_limit=(-0.1, 0.1),
                                        rotate_limit=(-0, 0))
        image, mask, boundary = randomHorizontalFlip(image, mask, boundary)
        image, mask, boundary = randomVerticleFlip(image, mask, boundary)
        image, mask, boundary = randomRotate90(image, mask, boundary)
    
    image = image.astype(np.float32) / 255.0 * 3.2 - 1.6  # -> (3, h, w)
    mask = mask.astype(np.float32)
    boundary = boundary.astype(np.float32)

    return image, mask, boundary



@TRANSFORMS.register_module()
class ALNetTransform(BaseTransform):
    """Add your transform

    Args:
        p (float): Probability of shifts. Default 0.5.
    """

    def __init__(self, input_size, mode):
        self.input_size = input_size
        self.mode = mode

    def transform(self, results):
        img = results['img']    # np.array (h, w, bgr)
        binary_mask = results['gt_seg_map']    # np.array (h, w), value belong to {0,1}

        boundary_path = results['seg_map_path'].replace('panoptic_seg_anns_coco', 'boundary_dir')
        boundary_mask = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
        boundary_mask[boundary_mask>0] = 1    # np.array (h, w), value belong to {0,1}

        datas = (img, binary_mask, boundary_mask)
        image, mask, boundary = transform_datas(datas, self.input_size, self.mode)
        results['img'] = image
        results['gt_seg_map'] = mask
        results['gt_boundary_mask'] = boundary
        return results


@TRANSFORMS.register_module()
class LoadBoundaryAnn(BaseTransform):
    """Add your transform

    Args:
        p (float): Probability of shifts. Default 0.5.
    """

    def __init__(self, scale=None):
        self.scale = scale

    def transform(self, results):
        boundary_path = results['seg_map_path'].replace('panoptic_seg_anns_coco', 'boundary_dir')
        boundary_mask = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
        boundary_mask[boundary_mask>0] = 1    # np.array (h, w), value belong to {0,1}
        if self.scale is not None:
            w,h = self.scale
            boundary_mask = cv2.resize(boundary_mask, (w, h))

        results['gt_boundary_mask'] = boundary_mask.astype(np.float32)
        return results