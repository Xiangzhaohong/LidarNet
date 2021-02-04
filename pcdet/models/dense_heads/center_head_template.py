import torch
import numpy as np
import torch.nn as nn
import math
from ...utils import center_utils, box_utils, common_utils
from ..model_utils import model_nms_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


def draw_heatmap(heatmap, boxes):
    raise NotImplementedError


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # gather的用法？？！！
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    # feat  :  (B, H, W, C)
    # ind   :  (B, K)
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _nms(heat, kernel=3):
    pad = (kernel - 1)//2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad
    )
    keep = (hmax == heat).float()
    return heat*keep


def _topk(scores, K=40):
    batch_size, cat, height, width = scores.size()
    # (batch_size, cat, K)
    topk_scores, topk_inds = torch.topk(scores.view(batch_size, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # (batch_size, K)
    topk_score, topk_ind = torch.topk(topk_scores.view(batch_size, -1), K)
    topk_clses = (topk_ind / K).int()   # from 0 start

    topk_inds = topk_inds.view(batch_size, -1)
    topk_inds = topk_inds.gather(1, topk_ind).view(batch_size, K)

    topk_ys = topk_ys.view(batch_size, -1)
    topk_ys = topk_ys.gather(1, topk_ind).view(batch_size, K)

    topk_xs = topk_xs.view(batch_size, -1)
    topk_xs = topk_xs.gather(1, topk_ind).view(batch_size, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _circle_nms(boxes, min_radius, post_max_size=83):
  """
        NMS according to center distance
  """
  keep = np.array(center_utils.circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]
  keep = torch.from_numpy(keep).long().to(boxes.device)
  return keep


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def gaussian_radius(det_size, min_overlap=0.5):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


# 这个生成高斯heatmap也不是特别巧妙
def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    radius_xy = np.array([radius, radius], dtype=np.float32)
    return fusion_heatmap(heatmap, gaussian, center, radius_xy, k)


def fusion_heatmap(heatmap, part_heatmap, center, radius, k=1):
    x, y = int(center[0]), int(center[1])
    radius_x, radius_y = int(radius[0]), int(radius[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = part_heatmap[radius_y - top:radius_y + bottom, radius_x - left:radius_x + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        # np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        torch.max(masked_heatmap, torch.from_numpy(masked_gaussian * k).type_as(masked_heatmap), out=masked_heatmap)
    return heatmap


def get_points_in_boxes3d(points, boxes3d):
    """
    Args:
        points: (num_points, 3 + C)
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps

    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) != 0]

    return points.numpy() if is_numpy else points


class CenterHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training, voxel_size):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.predict_boxes_when_training = predict_boxes_when_training
        self.voxel_size = voxel_size
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        self.forward_ret_dict = {}

    def AssignLabel(self, gt_boxes_classes, spatial_points=None):
        """
                Args:
                    gt_boxes_classes: (B, M, 8)
                    spatial_points: (B, 1, W, H)
                Returns:
                    target_heatmap: (B, class_num, W, H)
                    anno_box: (B, max_obj, 8/..)        # (offset_2, height_1, size_3, orientation_2/8)
                    {
                        mask = (batch_size, max_object)
                        ind = (batch_size, max_object)
                        cat = (batch_size, max_object)
                    }
        """
        # some param need to read from yaml
        feater_map_stride = int(self.model_cfg.TARGET_ASSIGNER_CONFIG.get('MAP_STRIDE', 1))
        # Optional ['2sin_cos', '8...']    -> 8... not implement
        encode_orientation_type = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('ORIENTATION_ENCODING_TYPE', '2sin_cos')
        # Optional ['umich_gaussian', 'car_shape', 'points_count']
        heatmap_type = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('HEATMAP_ENCODING_TYPE', 'umich_gaussian')
        gaussian_overlap = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('GAUSS_OVERLAP', 0.1)
        min_radius = int(self.model_cfg.TARGET_ASSIGNER_CONFIG.get('GAUSS_MIN_RADIUS', 2))
        auxiliary_reg = self.model_cfg.get('USE_AUXILIARY_REG', None)

        # here x->H, y->W
        mapW = int(self.grid_size[1]/feater_map_stride)
        mapH = int(self.grid_size[0]/feater_map_stride)
        # print('mapH: %d, mapW:%d' % (mapH, mapW))

        gt_boxes = gt_boxes_classes[:, :, :-1]
        gt_classes = gt_boxes_classes[:, :, -1]
        batch_size, max_object, obj_encode_num = gt_boxes.shape

        # Target here
        target_heatmap = torch.zeros(batch_size, self.num_class, mapW, mapH)
        if auxiliary_reg == 'point_counts':
            point_counts = torch.zeros(batch_size, self.num_class, mapW, mapH)
        elif auxiliary_reg == 'corner_cls':
            corner_cls = torch.zeros(batch_size, self.num_class, mapW, mapH)

        if encode_orientation_type == '2sin_cos':
            anno_box = torch.zeros(batch_size, max_object, 8)      # (offset:2, height:1, size:3, orientation:2 = 8)
        else:
            raise NotImplementedError('NOT REALIZE ALGORITHM!!')
        # torch.int64 is necessary for torch index !!
        mask = torch.zeros(batch_size, max_object, dtype=torch.int64)
        ind = torch.zeros(batch_size, max_object, dtype=torch.int64)
        cat = torch.zeros(batch_size, max_object, dtype=torch.int)

        example = {}
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.shape[0] - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            if spatial_points is not None:
                cur_spatial_points = spatial_points[k:k+1, 0:1, :, :]  # (W, H)
                avg_m = nn.AvgPool2d(feater_map_stride, stride=feater_map_stride)
                avg_out = avg_m(cur_spatial_points)
                cur_spatial_points = avg_out[0, 0, :, :].cpu()

            for i in range(cnt + 1):
                obj_box = cur_gt[i]
                obj_class = cur_gt_classes[i] - 1
                centerx, centery, centerz, dx, dy, dz, rot = obj_box.cpu().tolist()

                centerw = (centery - self.point_cloud_range[1]) / self.voxel_size[1] / feater_map_stride
                centerh = (centerx - self.point_cloud_range[0]) / self.voxel_size[0] / feater_map_stride
                centerw_int = int(centerw)
                centerh_int = int(centerh)

                # throw out not in range objects to avoid out of array area when gather
                if not (0 <= centerw_int < mapW and 0 <= centerh_int < mapH):
                    continue

                if heatmap_type == 'car_shape':
                    ###########################
                    # just like AFDet
                    # code need to optimize ...
                    ##########################
                    car_shape_w = int(dy / self.voxel_size[1] / feater_map_stride)
                    car_shape_h = int(dx / self.voxel_size[0] / feater_map_stride)
                    obj_heatmap = torch.zeros(2*car_shape_w + 1, 2*car_shape_h + 1)
                    for w in range(-car_shape_w, car_shape_w + 1):
                        for h in range(-car_shape_h, car_shape_h + 1):
                            distance = math.sqrt(math.pow(w, 2) + math.pow(h, 2))
                            temp_w = centerw_int + w
                            temp_h = centerh_int + h
                            if not (0 <= temp_w < mapW and 0 <= temp_h < mapH):
                                continue
                            if distance == 0:
                                obj_heatmap[w + car_shape_w, h + car_shape_h] = 1.0
                            elif distance == 1:
                                obj_heatmap[w + car_shape_w, h + car_shape_h] = 0.8
                            else:
                                obj_heatmap[w + car_shape_w, h + car_shape_h] = 1 / distance
                    ct = np.array([centerh_int, centerw_int], dtype=np.float32)
                    radius = np.array([car_shape_h, car_shape_w], dtype=np.float32)
                    fusion_heatmap(target_heatmap[k][obj_class], obj_heatmap.numpy(), ct, radius)
                elif heatmap_type == 'car_shape_real':
                    car_shape_w = int(dy / self.voxel_size[1] / feater_map_stride)
                    car_shape_h = int(dx / self.voxel_size[0] / feater_map_stride)
                    project_box = np.array([[centerh_int, centerw_int, 0, car_shape_h, car_shape_w, 0, rot]])
                    max_radius = math.ceil(math.sqrt(car_shape_h*car_shape_h + car_shape_w*car_shape_w)/2)
                    project_points = []
                    for hh in range(-max_radius, max_radius + 1):
                        for ww in range(-max_radius, max_radius + 1):
                            project_points.append([hh + centerh_int, ww + centerw_int, 0])
                    project_points = np.array(project_points)
                    project_points = get_points_in_boxes3d(project_points, project_box)

                    for nnn in range(project_points.shape[0]):
                        temp_h = int(project_points[nnn, 0])
                        temp_w = int(project_points[nnn, 1])
                        distance = math.sqrt(math.pow(temp_h - centerh_int, 2) + math.pow(temp_w - centerw_int, 2))
                        if not (0 <= temp_w < mapW and 0 <= temp_h < mapH):
                            continue
                        if distance == 0:
                            target_heatmap[k][obj_class][temp_w, temp_h] = max(target_heatmap[k][obj_class][temp_w, temp_h], 1.0)
                        elif distance == 1:
                            target_heatmap[k][obj_class][temp_w, temp_h] = max(target_heatmap[k][obj_class][temp_w, temp_h], 0.8)
                        else:
                            target_heatmap[k][obj_class][temp_w, temp_h] = max(target_heatmap[k][obj_class][temp_w, temp_h], 1/distance)
                elif heatmap_type == 'umich_gaussian':
                    ###########################
                    # just like CenterPoint
                    # 计算高斯半径这是用的栅格化的l和w
                    ##########################
                    radius = gaussian_radius((dy / self.voxel_size[1] / feater_map_stride, dx / self.voxel_size[0] / feater_map_stride),
                                             min_overlap=gaussian_overlap)
                    radius = max(min_radius, int(radius))
                    ct = np.array([centerh_int, centerw_int], dtype=np.float32)
                    draw_umich_gaussian(target_heatmap[k][obj_class], ct, radius)
                elif heatmap_type == 'points_count':
                    ###########################
                    # I think ....  (as OneNet, just want to verify)
                    # 根据点云的特征，如果回归的“中心点”是box所包含的栅格里点数最多的那个栅格，“特征中心点”
                    ##########################
                    if spatial_points is not None:
                        car_shape_w = int(dy / self.voxel_size[1] / feater_map_stride)
                        car_shape_h = int(dx / self.voxel_size[0] / feater_map_stride)

                        left, right = min(centerh_int, car_shape_h), min(mapH - centerh_int, car_shape_h + 1)
                        top, bottom = min(centerw_int, car_shape_w), min(mapW - centerw_int, car_shape_w + 1)

                        masked_heatmap = target_heatmap[k][obj_class][centerw_int-top:centerw_int+bottom, centerh_int-left:centerh_int+right]
                        masked_gaussian = cur_spatial_points[centerw_int-top:centerw_int+bottom, centerh_int-left:centerh_int+right]

                        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0 and masked_gaussian.max() > 0:  # TODO debug
                            # np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
                            top1_point, top1_ind = torch.topk(masked_gaussian.reshape(-1), 1)
                            centerw_int_temp = (top1_ind / (right + left) + centerw_int - top).int()
                            centerh_int_temp = (top1_ind % (bottom + top) + centerh_int - left).int()

                            centerw_int = centerw_int_temp
                            centerh_int = centerh_int_temp
                            masked_gaussian = masked_gaussian / masked_gaussian.max()
                            torch.max(masked_heatmap, masked_gaussian.type_as(masked_heatmap),
                                      out=masked_heatmap)
                        else:
                            continue
                    else:
                        raise Exception
                else:
                    raise NotImplementedError('NOT REALIZE ALGORITHM!!')

                if auxiliary_reg == 'point_counts_v1':
                    car_shape_w = int(dy / self.voxel_size[1] / feater_map_stride)
                    car_shape_h = int(dx / self.voxel_size[0] / feater_map_stride)
                    project_box = np.array([[centerh_int, centerw_int, 0, car_shape_h, car_shape_w, 0, rot]])
                    max_radius = math.ceil(math.sqrt(car_shape_h * car_shape_h + car_shape_w * car_shape_w) / 2)
                    project_points = []
                    for hh in range(-max_radius, max_radius + 1):
                        for ww in range(-max_radius, max_radius + 1):
                            if not (0 <= ww + centerw_int < mapW and 0 <= hh + centerh_int < mapH):
                                continue
                            project_points.append([hh + centerh_int, ww + centerw_int, 0,
                                                   cur_spatial_points[ww + centerw_int, hh + centerh_int]])
                    project_points = np.array(project_points)
                    project_points = get_points_in_boxes3d(project_points, project_box)
                    cur_max_count = max(project_points[:, 3])

                    if cur_max_count == 0:
                        # point_counts[k][obj_class][centerw_int, centerh_int] = 1.0
                        continue
                    else:
                        for nnn in range(project_points.shape[0]):
                            temp_h = int(project_points[nnn, 0])
                            temp_w = int(project_points[nnn, 1])
                            point_count_soft = project_points[nnn, 3] / cur_max_count
                            if not (0 <= temp_w < mapW and 0 <= temp_h < mapH):
                                continue
                            point_counts[k][obj_class][temp_w, temp_h] = float(point_count_soft)
                if auxiliary_reg == 'point_counts':
                    car_shape_w = int(dy / self.voxel_size[1] / feater_map_stride)
                    car_shape_h = int(dx / self.voxel_size[0] / feater_map_stride)

                    left, right = min(centerh_int, car_shape_h), min(mapH - centerh_int, car_shape_h + 1)
                    top, bottom = min(centerw_int, car_shape_w), min(mapW - centerw_int, car_shape_w + 1)
                    masked_pointcount = cur_spatial_points[centerw_int - top:centerw_int + bottom, centerh_int - left:centerh_int + right]
                    top1_point, top1_ind = torch.topk(masked_pointcount.reshape(-1), 1)
                    centerw_int_temp = (top1_ind / (right + left) + centerw_int - top).int()
                    centerh_int_temp = (top1_ind % (bottom + top) + centerh_int - left).int()

                    radius = gaussian_radius((dy / self.voxel_size[1] / feater_map_stride, dx / self.voxel_size[0] / feater_map_stride), min_overlap=gaussian_overlap)
                    radius = max(min_radius, int(radius))
                    ct = np.array([centerh_int_temp, centerw_int_temp], dtype=np.float32)
                    draw_umich_gaussian(point_counts[k][obj_class], ct, radius)
                if auxiliary_reg == 'corner_cls':
                    car_shape_w = dy / self.voxel_size[1] / feater_map_stride
                    car_shape_h = dx / self.voxel_size[0] / feater_map_stride
                    project_box = torch.tensor([centerh, centerw, 0, car_shape_h, car_shape_w, 0, rot]).float()
                    corner_points = box_utils.boxes_to_corners_3d(project_box.unsqueeze(0))
                    corner_points = corner_points[0, 0:4, 0:2]

                    radius = gaussian_radius((dy / self.voxel_size[1] / feater_map_stride, dx / self.voxel_size[0] / feater_map_stride),
                        min_overlap=gaussian_overlap)
                    radius = max(min_radius, int(radius))
                    for co in range(4):
                        ct = np.array([corner_points[co, 0].int(), corner_points[co, 1].int()], dtype=np.float32)
                        draw_umich_gaussian(corner_cls[k][obj_class], ct, radius)
                # here cls is start from 0, and our predict && gt is start from 1
                cat[k][i] = obj_class
                mask[k][i] = 1
                # check is error 为了匹配后面的WxH，需要仔细对应的维度
                ind[k][i] = centerw_int*mapH + centerh_int   # 后面gather用

                if encode_orientation_type == '2sin_cos':
                    anno_box[k][i] = anno_box.new_tensor([centerh - centerh_int, centerw - centerw_int, centerz,
                                                          dx, dy, dz, math.sin(rot), math.cos(rot)])
                else:
                    raise NotImplementedError

        example.update({'hm': target_heatmap.cuda(), 'anno_box': anno_box.cuda(), 'ind': ind.cuda(), 'mask': mask.cuda(), 'cat': cat.cuda()})
        if auxiliary_reg == 'point_counts':
            example.update({'point_counts': point_counts.cuda()})
        if auxiliary_reg == 'corner_cls':
            example.update({'corner_cls': corner_cls.cuda()})
        return example

    def get_hm_loss(self):
        # 需要sigmoid, 不过这里clamp ?
        # pred_hm = _sigmoid(self.forward_ret_dict['heatmap'])   # (B, W, H, class_num)
        pred_hm = self.forward_ret_dict['heatmap']  # (B, W, H, class_num)
        gt_hm = self.forward_ret_dict['hm']     # (B, class_num, W, H)
        gt_hm = gt_hm.permute(0, 2, 3, 1).contiguous()
        hm_loss = center_utils.Center_FocalLoss(pred_hm, gt_hm)
        # 对batch和obj_num的归一化在Center_FocalLoss里
        hm_loss = hm_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']
        tb_dict = {
            'hm_loss': hm_loss.item()
        }
        return hm_loss, tb_dict

    def get_offset_loss(self):
        off_radius = self.model_cfg.TARGET_ASSIGNER_CONFIG.OFFSET_RADIUS
        pred_offset = self.forward_ret_dict['offset']     # (B, W, H, 2)
        gt_offset = self.forward_ret_dict['anno_box'][:, :, 0:2]    # (B, max_obj, 2)
        mask = self.forward_ret_dict['mask']    # (batch_size, max_object)
        ind = self.forward_ret_dict['ind']      # (batch_size, max_object)
        # print(ind.shape)
        # print(ind.max())
        # print(ind.min())
        if off_radius == 0:
            pred_offset = pred_offset.view(pred_offset.size(0), -1, pred_offset.size(-1))
            dim = pred_offset.size(-1)
            ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
            pred_offset = pred_offset.gather(1, ind)    # !!!!
            # mask = mask.unsqueeze(2).expand_as(pred_offset)
            # pred_offset = pred_offset[mask]
            # pred_offset = pred_offset.view(-1, dim)
            offset_loss = center_utils.Center_RegLoss(pred_offset, gt_offset, mask)
        else:
            raise NotImplementedError('should like afdet paper -> have radius')

        offset_loss = offset_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['offset_weight']
        # print(offset_loss)
        tb_dict = {
            'offset_loss': offset_loss.item()
        }
        return offset_loss, tb_dict

    def get_height_loss(self, LOSS_FUNC=None):
        pred_height = self.forward_ret_dict['height']  # (B, W, H, 2)
        gt_height = self.forward_ret_dict['anno_box'][:, :, 2:3]  # (B, max_obj, 2)
        mask = self.forward_ret_dict['mask']  # (batch_size, max_object)
        ind = self.forward_ret_dict['ind']  # (batch_size, max_object)

        pred_height = pred_height.view(pred_height.size(0), -1, pred_height.size(-1))
        dim = pred_height.size(-1)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        pred_height = pred_height.gather(1, ind)  # !!!!

        if LOSS_FUNC is None:
            height_loss = center_utils.Center_RegLoss(pred_height, gt_height, mask)
        else:
            height_loss = LOSS_FUNC(pred_height, gt_height, mask)

        height_loss = height_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['height_weight']
        tb_dict = {
            'height_loss': height_loss.item()
        }
        return height_loss, tb_dict

    def get_size_loss(self, LOSS_FUNC=None):
        pred_size = self.forward_ret_dict['size']  # (B, W, H, 2)
        gt_size = self.forward_ret_dict['anno_box'][:, :, 3:6]  # (B, max_obj, 2)
        mask = self.forward_ret_dict['mask']  # (batch_size, max_object)
        ind = self.forward_ret_dict['ind']  # (batch_size, max_object)

        pred_size = pred_size.view(pred_size.size(0), -1, pred_size.size(-1))
        dim = pred_size.size(-1)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        pred_size = pred_size.gather(1, ind)  # !!!!

        if LOSS_FUNC is None:
            size_loss = center_utils.Center_RegLoss(pred_size, gt_size, mask)
        else:
            size_loss = LOSS_FUNC(pred_size, gt_size, mask)

        size_loss = size_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['size_weight']
        tb_dict = {
            'size_loss': size_loss.item()
        }
        return size_loss, tb_dict

    def get_orientation_loss(self):
        orientation_encode_type = self.model_cfg.get('ORIENTATION_ENCODING_TYPE', '2sin_cos')

        pred_orientation = self.forward_ret_dict['orientation']  # (B, W, H, 2)
        mask = self.forward_ret_dict['mask']  # (batch_size, max_object)
        ind = self.forward_ret_dict['ind']  # (batch_size, max_object)

        pred_orientation = pred_orientation.view(pred_orientation.size(0), -1, pred_orientation.size(-1))
        dim = pred_orientation.size(-1)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        pred_orientation = pred_orientation.gather(1, ind)  # !!!!

        if orientation_encode_type == '2sin_cos':
            gt_orientation = self.forward_ret_dict['anno_box'][:, :, 6:8]  # (B, max_obj, 2)
            orientation_loss = center_utils.Center_RegLoss(pred_orientation, gt_orientation, mask)
        else:
            raise NotImplementedError('NOT REALIZE ALGORITHM!!')

        orientation_loss = orientation_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['ori_weight']
        tb_dict = {
            'orientation_loss': orientation_loss.item()
        }
        return orientation_loss, tb_dict

    def get_loss(self):
        hm_loss, tb_dict = self.get_hm_loss()
        offset_loss, tb_dict_offset = self.get_offset_loss()

        if self.model_cfg['LOSS_CONFIG'].get('USE_BalancedL1Loss', False):
            BLoss = center_utils.BalancedL1Loss(alpha=0.5, gamma=1.5)
            height_loss, tb_dict_height = self.get_height_loss(LOSS_FUNC=BLoss)
            size_loss, tb_dict_size = self.get_size_loss(LOSS_FUNC=BLoss)
        else:
            height_loss, tb_dict_height = self.get_height_loss()
            size_loss, tb_dict_size = self.get_size_loss()

        orientation_loss, tb_dict_orientation = self.get_orientation_loss()

        tb_dict.update(tb_dict_offset)
        tb_dict.update(tb_dict_height)
        tb_dict.update(tb_dict_size)
        tb_dict.update(tb_dict_orientation)

        rpn_loss = hm_loss + offset_loss + height_loss + size_loss + orientation_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, pred_hm, pred_offset, pred_height, pred_size, pred_ori):
        """
            Args:
            Returns:
                pred_dicts: (B, num_boxes, num_classes)
                recall_dict: (B, num_boxes, 7+C)
        """
        # 这里一般不会和上面计算loss同时运行(除非roi头),所以可以不考虑上面loss时对hm offset...的操作
        # (就是tensor那些操作涉及的共享内存/梯度传播 问题)
        orientation_encode_type = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('ORIENTATION_ENCODING_TYPE', '2sin_cos')
        feater_map_stride = int(self.model_cfg.TARGET_ASSIGNER_CONFIG.get('MAP_STRIDE', 1))
        use_maxpool = self.model_cfg.POST_CONFIG.get('USE_MAXPOOL', False)
        use_circle_nms = self.model_cfg.POST_CONFIG.get('USE_CIRCLE_NMS', False)
        use_iou_nms = self.model_cfg.POST_CONFIG.get('USE_IOU_NMS', False)
        circle_nms_min_radius = self.model_cfg.POST_CONFIG.get('MIN_RADIUS', None)
        max_per_img = self.model_cfg.POST_CONFIG.get('MAX_PRE_IMG', 500)
        post_max_size = self.model_cfg.POST_CONFIG.get('MAX_POST', 83)
        score_threshold = self.model_cfg.POST_CONFIG.get('SCORE_THRESHOLD', 0)

        if not isinstance(pred_hm, list):
            pred_hm = [pred_hm]
            pred_offset = [pred_offset]
            pred_height = [pred_height]
            pred_size = [pred_size]
            pred_ori = [pred_ori]

        recall_dict = {}
        pred_dicts = []

        for idx in range(len(pred_hm)):
            cur_pred_hm = pred_hm[idx]
            cur_pred_offset = pred_offset[idx]
            cur_pred_height = pred_height[idx]
            cur_pred_size = pred_size[idx]
            cur_pred_ori = pred_ori[idx]

            batch_size = cur_pred_hm.size(0)
            cur_pred_hm = cur_pred_hm.permute(0, 3, 1, 2).contiguous()  # (B, class_num, W, H)
            # maxpool_2D input size (N,C,H,W) , output (N, C, H_{out}, W_{out})
            if use_maxpool:
                cur_pred_hm = _nms(heat=cur_pred_hm, kernel=3)
            topk_score, topk_inds, topk_clses, topk_ys, topk_xs = _topk(cur_pred_hm, max_per_img)

            xs_key = topk_xs.view(batch_size, max_per_img, 1) * self.voxel_size[0] * feater_map_stride + self.point_cloud_range[0]
            ys_key = topk_ys.view(batch_size, max_per_img, 1) * self.voxel_size[1] * feater_map_stride + self.point_cloud_range[1]
            if cur_pred_offset is not None:
                cur_pred_offset = _transpose_and_gather_feat(cur_pred_offset, topk_inds)
                cur_pred_offset = cur_pred_offset.view(batch_size, max_per_img, 2)
                # offset是直接加在这的，就是加在栅格坐标上
                xs = topk_xs.view(batch_size, max_per_img, 1) + cur_pred_offset[:, :, 0:1]
                ys = topk_ys.view(batch_size, max_per_img, 1) + cur_pred_offset[:, :, 1:2]
            else:
                xs = topk_xs.view(batch_size, max_per_img, 1) + 0.5
                ys = topk_ys.view(batch_size, max_per_img, 1) + 0.5

            if orientation_encode_type == '2sin_cos':
                rots = _transpose_and_gather_feat(cur_pred_ori[:, :, :, 0:1], topk_inds)
                rots = rots.view(batch_size, max_per_img, 1)
                rotc = _transpose_and_gather_feat(cur_pred_ori[:, :, :, 1:2], topk_inds)
                rotc = rotc.view(batch_size, max_per_img, 1)

                rot = torch.atan2(rots, rotc)
            else:
                raise Exception('not code!')

            height = _transpose_and_gather_feat(cur_pred_height, topk_inds)
            height = height.view(batch_size, max_per_img, 1)

            dim = _transpose_and_gather_feat(cur_pred_size, topk_inds)
            dim = dim.view(batch_size, max_per_img, 3)

            clses = topk_clses.view(batch_size, max_per_img)
            scores = topk_score.view(batch_size, max_per_img)

            xs = xs.view(batch_size, max_per_img, 1) * self.voxel_size[0] * feater_map_stride + self.point_cloud_range[0]
            ys = ys.view(batch_size, max_per_img, 1) * self.voxel_size[1] * feater_map_stride + self.point_cloud_range[1]

            # get final !!
            final_box_preds = torch.cat(
                [xs, ys, height, dim, rot], dim=2
            )
            final_keypoint_preds = torch.cat([xs_key, ys_key, height], dim=2)
            final_scores = scores
            final_class = clses

            # max_pool_nms是在topk之前用；；；circle_nms是在topk后用
            if score_threshold is not None:
                thresh_mask = final_scores > score_threshold

            for i in range(batch_size):
                if score_threshold:
                    boxes3d = final_box_preds[i, thresh_mask[i]]
                    scores = final_scores[i, thresh_mask[i]]
                    labels = final_class[i, thresh_mask[i]]
                    keypoints = final_keypoint_preds[i, thresh_mask[i]]
                else:
                    boxes3d = final_box_preds[i]
                    scores = final_scores[i]
                    labels = final_class[i]
                    keypoints = final_keypoint_preds[i]

                if self.use_multihead:
                    cur_label_mapping = self.rpn_heads[idx].head_label_indices
                    labels = cur_label_mapping[labels.long()]
                else:
                    labels = labels + 1

                if use_circle_nms:
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1).detach()
                    keep = _circle_nms(boxes, min_radius=circle_nms_min_radius[idx], post_max_size=post_max_size)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    keypoints = keypoints[keep]
                elif use_iou_nms and self.model_cfg.POST_CONFIG.NMS_CONFIG.get('MULTI_CLASSES_NMS', False):
                    # use normal nms
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=scores, box_preds=boxes3d,
                        nms_config=self.model_cfg.POST_CONFIG.NMS_CONFIG,
                        score_thresh=None
                    )
                    boxes3d = boxes3d[selected]
                    scores = scores[selected]
                    labels = labels[selected]
                    keypoints = keypoints[selected]

                record_dict = {
                    'pred_boxes': boxes3d,
                    'pred_scores': scores,
                    'pred_labels': labels,
                    'pred_keypoints': keypoints
                }

                if idx == 0:
                    pred_dicts.append(record_dict)
                else:
                    pred_dicts[i]['pred_boxes'] = torch.cat([pred_dicts[i]['pred_boxes'], record_dict['pred_boxes']], dim=0)
                    pred_dicts[i]['pred_scores'] = torch.cat([pred_dicts[i]['pred_scores'], record_dict['pred_scores']], dim=0)
                    pred_dicts[i]['pred_labels'] = torch.cat([pred_dicts[i]['pred_labels'], record_dict['pred_labels']], dim=0)
                    pred_dicts[i]['pred_keypoints'] = torch.cat([pred_dicts[i]['pred_keypoints'], record_dict['pred_keypoints']], dim=0)

        if use_iou_nms and not self.model_cfg.POST_CONFIG.NMS_CONFIG.get('MULTI_CLASSES_NMS', False):
            # use normal nms
            batch_size = pred_hm[0].size(0)
            for i in range(batch_size):
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=pred_dicts[i]['pred_scores'], box_preds=pred_dicts[i]['pred_boxes'],
                    nms_config=self.model_cfg.POST_CONFIG.NMS_CONFIG,
                    score_thresh=None
                )
                pred_dicts[i]['pred_boxes'] = pred_dicts[i]['pred_boxes'][selected]
                pred_dicts[i]['pred_scores'] = pred_dicts[i]['pred_scores'][selected]
                pred_dicts[i]['pred_labels'] = pred_dicts[i]['pred_labels'][selected]
                pred_dicts[i]['pred_keypoints'] = pred_dicts[i]['pred_keypoints'][selected]

        return pred_dicts, recall_dict

    def forward(self, **kwargs):
        raise NotImplementedError
