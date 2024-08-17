import cv2
import numpy as np
import torch
from mmdet.evaluation.metrics import SemSegMetric,CocoPanopticMetric
from prettytable import PrettyTable
import torch
from skimage.measure import label
from scipy.ndimage import binary_dilation
from mmengine.logging import MMLogger
from scipy.optimize import linear_sum_assignment


def find_connect(binary_mask: torch.Tensor, dilation = False):

    image_mask_np = binary_mask.cpu().numpy().astype(np.uint8)
    # labels: 0 mean background, 1...n mean different connection region
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mask_np, connectivity=8)
    if dilation and np.sum(labels) > 0:
        iid = np.unique(labels)
        for i in iid[1:]:
            mask = labels == i
            dilation_mask = binary_dilation(mask)
            labels[dilation_mask] = i
    
    return labels

class InstMetric:
    def __init__(self, indicator) -> None:
        '''
        Args:
            - indicator: 'all' or one of ['AJI','AJI_plus','Dice','PQ']
        '''
        super().__init__()

        support_indicator = ['all', 'AJI','AJI_plus','Dice','PQ']

        assert indicator in support_indicator, \
            f'the indicator must be in {support_indicator}'

        self.indicator = indicator

        if indicator == 'all':
            self.metrics = dict(
                AJI = None, AJI_plus = None, IoU = None, Dice = None,
                PQ = None, SQ = None, DQ = None,
            )

            self.all_metrics_name = [
                'aji_scores', 'aji_plus_scores', 'iou_scores', 'dice_scores',
                'PQ_scores', 'SQ_scores', 'DQ_scores',
            ]
        else:
            if indicator == 'AJI':
                self.metrics = dict(AJI = None)
                self.all_metrics_name = ['aji_scores']
            if indicator == 'AJI_plus':
                self.metrics = dict(AJI_plus = None)
                self.all_metrics_name = ['aji_plus_scores']
            if indicator == 'Dice':
                self.metrics = dict(IoU = None, Dice = None)
                self.all_metrics_name = ['iou_scores', 'dice_scores']
            if indicator == 'PQ':
                self.metrics = dict(PQ = None, SQ = None, DQ = None)
                self.all_metrics_name = ['PQ_scores', 'SQ_scores', 'DQ_scores']

        self.aji_scores = []
        self.aji_plus_scores = []
        self.iou_scores = []
        self.dice_scores = []
        self.PQ_scores = []
        self.SQ_scores = []
        self.DQ_scores = []
    
    def find_connect(self, binary_mask: torch.Tensor, dilation = False):
        return find_connect(binary_mask, dilation)

    def AJI_fast(self, gt_mask, pred_mask):
        '''
        Requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
        not [2, 3, 6, 10].
        Args:
            - gt_mask: (np.array), shape is (h,w), pixel value > 0 means instance IDs, 0 mean background
            - pred_mask: (np.array), shape is (h,w), pixel value means same as gt_mask
        '''
        gs, g_areas = np.unique(gt_mask, return_counts=True)
        assert np.all(gs == np.arange(len(gs)))
        ss, s_areas = np.unique(pred_mask, return_counts=True)
        assert np.all(ss == np.arange(len(ss)))

        i_idx, i_cnt = np.unique(np.concatenate([gt_mask.reshape(1, -1), pred_mask.reshape(1, -1)]),
                                return_counts=True, axis=1)
        i_arr = np.zeros(shape=(len(gs), len(ss)), dtype=np.int32)
        i_arr[i_idx[0], i_idx[1]] += i_cnt
        u_arr = g_areas.reshape(-1, 1) + s_areas.reshape(1, -1) - i_arr
        iou_arr = 1.0 * i_arr / u_arr

        i_arr = i_arr[1:, 1:]
        u_arr = u_arr[1:, 1:]
        iou_arr = iou_arr[1:, 1:]

        # no pred inst
        if iou_arr.shape[1] == 0:
            return 0.
        j = np.argmax(iou_arr, axis=1)

        c = np.sum(i_arr[np.arange(len(gs) - 1), j])
        u = np.sum(u_arr[np.arange(len(gs) - 1), j])
        used = np.zeros(shape=(len(ss) - 1), dtype=np.int32)
        used[j] = 1
        u += (np.sum(s_areas[1:] * (1 - used)))
        return 1.0 * c / u

    def get_fast_aji_plus(self, true, pred):
        """AJI+, an AJI version with maximal unique pairing to obtain overall intersecion.
        Every prediction instance is paired with at most 1 GT instance (1 to 1) mapping, unlike AJI 
        where a prediction instance can be paired against many GT instances (1 to many).
        Remaining unpaired GT and Prediction instances will be added to the overall union.
        The 1 to 1 mapping prevents AJI's over-penalisation from happening.

        Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
        not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
        effect on the result.

        """
        true = np.copy(true)  # ? do we need this
        pred = np.copy(pred)
        true_id_list = list(np.unique(true))
        pred_id_list = list(np.unique(pred))

        true_masks = [
            None,
        ]
        for t in true_id_list[1:]:
            t_mask = np.array(true == t, np.uint8)
            true_masks.append(t_mask)

        pred_masks = [
            None,
        ]
        for p in pred_id_list[1:]:
            p_mask = np.array(pred == p, np.uint8)
            pred_masks.append(p_mask)

        # prefill with value
        pairwise_inter = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )
        pairwise_union = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )

        # caching pairwise
        for true_id in true_id_list[1:]:  # 0-th is background
            t_mask = true_masks[true_id]
            pred_true_overlap = pred[t_mask > 0]
            pred_true_overlap_id = np.unique(pred_true_overlap)
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:
                if pred_id == 0:  # ignore
                    continue  # overlaping background
                p_mask = pred_masks[pred_id]
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                pairwise_inter[true_id - 1, pred_id - 1] = inter
                pairwise_union[true_id - 1, pred_id - 1] = total - inter
        #
        pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
        #### Munkres pairing to find maximal unique pairing
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]
        # now select all those paired with iou != 0.0 i.e have intersection
        paired_true = paired_true[paired_iou > 0.0]
        paired_pred = paired_pred[paired_iou > 0.0]
        paired_inter = pairwise_inter[paired_true, paired_pred]
        paired_union = pairwise_union[paired_true, paired_pred]
        paired_true = list(paired_true + 1)  # index to instance ID
        paired_pred = list(paired_pred + 1)
        overall_inter = paired_inter.sum()
        overall_union = paired_union.sum()
        # add all unpaired GT and Prediction into the union
        unpaired_true = np.array(
            [idx for idx in true_id_list[1:] if idx not in paired_true]
        )
        unpaired_pred = np.array(
            [idx for idx in pred_id_list[1:] if idx not in paired_pred]
        )
        for true_id in unpaired_true:
            overall_union += true_masks[true_id].sum()
        for pred_id in unpaired_pred:
            overall_union += pred_masks[pred_id].sum()
        #
        aji_score = overall_inter / (overall_union+1e-6)
        return aji_score

    def inst_iou_dice(self, gt, pred):
        """ Compute the object-level metrics between predicted and
        groundtruth: dice, iou """

        # get connected components
        pred_labeled = label(pred, connectivity=2)
        Ns = len(np.unique(pred_labeled)) - 1
        gt_labeled = label(gt, connectivity=2)
        Ng = len(np.unique(gt_labeled)) - 1

        # --- compute dice, iou, hausdorff --- #
        pred_objs_area = np.sum(pred_labeled>0)  # total area of objects in image
        gt_objs_area = np.sum(gt_labeled>0)  # total area of objects in groundtruth gt

        # compute how well groundtruth object overlaps its segmented object
        dice_g = 0.0
        iou_g = 0.0
        for i in range(1, Ng + 1):
            gt_i = np.where(gt_labeled == i, 1, 0)
            overlap_parts = gt_i * pred_labeled
            # get intersection objects numbers in image
            obj_no = np.unique(overlap_parts)
            obj_no = obj_no[obj_no != 0]
            gamma_i = float(np.sum(gt_i)) / gt_objs_area
            if obj_no.size == 0:   # no intersection object
                dice_i = 0
                iou_i = 0
            else:
                # find max overlap object
                obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
                seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
                pred_i = np.where(pred_labeled == seg_obj, 1, 0)  # segmented object
                overlap_area = np.max(obj_areas)  # overlap area
                dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
                iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)
            dice_g += gamma_i * dice_i
            iou_g += gamma_i * iou_i
            
        # compute how well segmented object overlaps its groundtruth object
        dice_s = 0.0
        iou_s = 0.0
        for j in range(1, Ns + 1):
            pred_j = np.where(pred_labeled == j, 1, 0)
            overlap_parts = pred_j * gt_labeled
            # get intersection objects number in gt
            obj_no = np.unique(overlap_parts)
            obj_no = obj_no[obj_no != 0]
            sigma_j = float(np.sum(pred_j)) / pred_objs_area
            # no intersection object
            if obj_no.size == 0:
                dice_j = 0
                iou_j = 0
            else:
                # find max overlap gt
                gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
                gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
                gt_j = np.where(gt_labeled == gt_obj, 1, 0)  # groundtruth object

                overlap_area = np.max(gt_areas)  # overlap area
                dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
                iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)
            dice_s += sigma_j * dice_j
            iou_s += sigma_j * iou_j

        return (iou_g + iou_s) / 2, (dice_g + dice_s) / 2

    def get_fast_pq(self, true, pred, match_iou=0.5):
        """`match_iou` is the IoU threshold level to determine the pairing between
        GT instances `p` and prediction instances `g`. `p` and `g` is a pair
        if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
        (1 prediction instance to 1 GT instance mapping).

        If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
        in bipartite graphs) is caculated to find the maximal amount of unique pairing. 

        If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
        the number of pairs is also maximal.    
        
        Fast computation requires instance IDs are in contiguous orderding 
        i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
        and `by_size` flag has no effect on the result.

        Returns:
            [dq, sq, pq]: measurement statistic

            [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                        pairing information to perform measurement
                        
        """
        assert match_iou >= 0.0, "Cant' be negative"

        true = np.copy(true)
        pred = np.copy(pred)
        true_id_list = list(np.unique(true))
        pred_id_list = list(np.unique(pred))

        true_masks = [
            None,
        ]
        for t in true_id_list[1:]:
            t_mask = np.array(true == t, np.uint8)
            true_masks.append(t_mask)

        pred_masks = [
            None,
        ]
        for p in pred_id_list[1:]:
            p_mask = np.array(pred == p, np.uint8)
            pred_masks.append(p_mask)

        # prefill with value
        pairwise_iou = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )

        # caching pairwise iou
        for true_id in true_id_list[1:]:  # 0-th is background
            t_mask = true_masks[true_id]
            pred_true_overlap = pred[t_mask > 0]
            pred_true_overlap_id = np.unique(pred_true_overlap)
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:
                if pred_id == 0:  # ignore
                    continue  # overlaping background
                p_mask = pred_masks[pred_id]
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                iou = inter / (total - inter)
                pairwise_iou[true_id - 1, pred_id - 1] = iou
        #
        if match_iou >= 0.5:
            paired_iou = pairwise_iou[pairwise_iou > match_iou]
            pairwise_iou[pairwise_iou <= match_iou] = 0.0
            paired_true, paired_pred = np.nonzero(pairwise_iou)
            paired_iou = pairwise_iou[paired_true, paired_pred]
            paired_true += 1  # index is instance id - 1
            paired_pred += 1  # hence return back to original
        else:  # * Exhaustive maximal unique pairing
            #### Munkres pairing with scipy library
            # the algorithm return (row indices, matched column indices)
            # if there is multiple same cost in a row, index of first occurence
            # is return, thus the unique pairing is ensure
            # inverse pair to get high IoU as minimum
            paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
            ### extract the paired cost and remove invalid pair
            paired_iou = pairwise_iou[paired_true, paired_pred]

            # now select those above threshold level
            # paired with iou = 0.0 i.e no intersection => FP or FN
            paired_true = list(paired_true[paired_iou > match_iou] + 1)
            paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
            paired_iou = paired_iou[paired_iou > match_iou]

        # get the actual FP and FN
        unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
        unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
        # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

        #
        tp = len(paired_true)
        fp = len(unpaired_pred)
        fn = len(unpaired_true)
        # get the F1-score i.e DQ
        dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
        # get the SQ, no paired has 0 iou so not impact
        sq = paired_iou.sum() / (tp + 1.0e-6)

        return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]

    
    def process(self, gt_mask, pred_mask):
        if not isinstance(gt_mask, np.ndarray):
            gt_mask = np.array(gt_mask)

        if self.indicator == 'AJI' or self.indicator == 'all':
            aji_score = self.AJI_fast(gt_mask, pred_mask)
            self.aji_scores.append(aji_score)
        if self.indicator == 'AJI_plus'or self.indicator == 'all':
            aji_plus_score = self.get_fast_aji_plus(gt_mask, pred_mask)
            self.aji_plus_scores.append(aji_plus_score)
        if self.indicator == 'Dice'or self.indicator == 'all':
            iou_score,dice_score = self.inst_iou_dice(gt_mask, pred_mask)
            self.iou_scores.append(iou_score)
            self.dice_scores.append(dice_score)
        if self.indicator == 'PQ'or self.indicator == 'all':
            # PQs: dq, sq, pq
            PQs,_ = self.get_fast_pq(gt_mask, pred_mask)
            self.PQ_scores.append(PQs[2])
            self.SQ_scores.append(PQs[1])
            self.DQ_scores.append(PQs[0])
    
    def evaluate(self, total_cnts):
            logger: MMLogger = MMLogger.get_current_instance()
            table_data = PrettyTable()
            for key_name, value_name in zip(self.metrics.keys(), self.all_metrics_name):
                values = getattr(self, value_name)
                average_value = np.sum(np.array(values)) / total_cnts
                setattr(self, value_name, [])
                self.metrics[key_name] = average_value
            
            for key,value in self.metrics.items():
                table_data.add_column(key,[f'{value:.4f}'])
            logger.info('\n' + table_data.get_string())

            return self.metrics


def get_metrics(types:list[str], metainfo, restinfo: dict = None, inst_indicator = None):
    '''can choice metric: ['pixel', 'inst', 'panoptic']
        pixel:  semantic_evaluator(mIoU, mFscore)
        inst: panoptic_evaluator(bPQ, mPQ...), AJI_evaluator
    '''
    evaluators = {}

    if 'pixel' in types:
        evaluators['semantic_evaluator'] = SemSegMetric(iou_metrics=['mIoU', 'mFscore', 'mDice'])
        evaluators['semantic_evaluator'].dataset_meta = metainfo

    if 'panoptic' in types:
        evaluators['panoptic_evaluator'] = CocoPanopticMetric(
            ann_file = restinfo['panoptic_ann_file'],
            seg_prefix = restinfo['seg_prefix'],
            classwise = True
        )
        evaluators['panoptic_evaluator'].dataset_meta = metainfo
    
    if 'inst' in types :
        evaluators['inst_evaluator'] = InstMetric(indicator=inst_indicator)

    return evaluators

