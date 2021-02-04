import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from ..backbones_2d import BaseBEVBackbone

from .center_head_template import CenterHeadTemplate, _sigmoid
from ...utils import weight_init_utils, center_utils, onenet_utils, loss_utils, box_utils
from ...ops.iou3d_nms import iou3d_nms_utils


class SingleHead(BaseBEVBackbone):
    def __init__(self, model_cfg, input_channels, num_class, rpn_head_cfg=None,
                 head_label_indices=None, separate_reg_config=None, init_bias=-2.19):
        super().__init__(rpn_head_cfg, input_channels)
        self.num_class = num_class
        self.model_cfg = model_cfg
        self.separate_reg_config = separate_reg_config
        self.register_buffer('head_label_indices', head_label_indices)

        if separate_reg_config is not None:
            num_middle_filter = separate_reg_config.NUM_MIDDLE_FILTER
            num_middle_kernel = separate_reg_config.NUM_MIDDLE_KERNEL
            reg_list = separate_reg_config.REG_LIST
            final_kernel = separate_reg_config.FINAL_KERNEL

            self.box_maps = nn.ModuleList()
            self.box_map_names = []

            heatmap_conv_list = []
            c_in = input_channels
            for middle_filter, middle_kenel in zip(num_middle_filter, num_middle_kernel):
                heatmap_conv_list.extend([
                    nn.Conv2d(c_in, middle_filter, kernel_size=middle_kenel, stride=1, padding=middle_kenel // 2,
                              bias=False),
                    nn.BatchNorm2d(middle_filter),
                    nn.ReLU()
                ])
                c_in = middle_filter
            heatmap_conv_list.extend([
                nn.Conv2d(c_in, self.num_class, kernel_size=final_kernel, stride=1, padding=final_kernel // 2)
            ])
            self.heat_map = nn.Sequential(*heatmap_conv_list)

            if self.training and self.model_cfg.get('USE_AUXILIARY_REG', None) == 'point_counts':
                AUX_PointCount_conv_list = []
                c_in = input_channels
                for middle_filter, middle_kenel in zip(num_middle_filter, num_middle_kernel):
                    AUX_PointCount_conv_list.extend([
                        nn.Conv2d(c_in, middle_filter, kernel_size=middle_kenel, stride=1, padding=middle_kenel // 2, bias=False),
                        nn.BatchNorm2d(middle_filter),
                        nn.ReLU()
                    ])
                    c_in = middle_filter
                AUX_PointCount_conv_list.extend([nn.Conv2d(c_in, self.num_class, kernel_size=final_kernel, stride=1, padding=final_kernel // 2)])
                self.AUX_PointCount_map = nn.Sequential(*AUX_PointCount_conv_list)
            elif self.training and self.model_cfg.get('USE_AUXILIARY_REG', None) == 'corner_cls':
                AUX_corner_conv_list = []
                c_in = input_channels
                for middle_filter, middle_kenel in zip(num_middle_filter, num_middle_kernel):
                    AUX_corner_conv_list.extend([
                        nn.Conv2d(c_in, middle_filter, kernel_size=middle_kenel, stride=1, padding=middle_kenel // 2, bias=False),
                        nn.BatchNorm2d(middle_filter),
                        nn.ReLU()
                    ])
                    c_in = middle_filter
                AUX_corner_conv_list.extend([nn.Conv2d(c_in, self.num_class, kernel_size=final_kernel, stride=1, padding=final_kernel // 2)])
                self.AUX_corner_map = nn.Sequential(*AUX_corner_conv_list)

            for reg_config in reg_list:
                reg_name, reg_channel = reg_config.split(':')
                cur_conv_list = []
                c_in = input_channels
                for middle_filter, middle_kenel in zip(num_middle_filter, num_middle_kernel):
                    cur_conv_list.extend([
                        nn.Conv2d(c_in, middle_filter, kernel_size=middle_kenel, stride=1, padding=middle_kenel // 2,
                                  bias=False),
                        nn.BatchNorm2d(middle_filter),
                        nn.ReLU()
                    ])
                    c_in = middle_filter
                cur_conv_list.extend([
                    nn.Conv2d(c_in, int(reg_channel), kernel_size=final_kernel, stride=1, padding=final_kernel // 2)
                ])
                self.box_maps.append(nn.Sequential(*cur_conv_list))
                self.box_map_names.append(f'{reg_name}')

            # init as center_point
            # 这里对hm头的初始化！！！
            self.heat_map[-1].bias.data.fill_(init_bias)
            if self.training and self.model_cfg.get('USE_AUXILIARY_REG', None) == 'point_counts':
                self.AUX_PointCount_map[-1].bias.data.fill_(init_bias)
            if self.training and self.model_cfg.get('USE_AUXILIARY_REG', None) == 'corner_cls':
                self.AUX_corner_map[-1].bias.data.fill_(init_bias)

            for i in range(len(self.box_maps)):
                for m in self.box_maps[i].modules():
                    if isinstance(m, nn.Conv2d):
                        weight_init_utils.kaiming_init(m)
        else:
            # TODO
            # 应该heatmap头单独，然后所有类共享一个regmap？
            raise NotImplementedError

    def forward(self, spatial_features_2d):
        pred_dict = {}
        # TODO
        # for RPN, such as multi different size of feature map
        spatial_features_2d = super().forward({'spatial_features': spatial_features_2d})['spatial_features_2d']

        heatmap_pred = _sigmoid(self.heat_map(spatial_features_2d))
        pred_dict['heatmap'] = heatmap_pred.permute(0, 2, 3, 1).contiguous()

        if self.training and self.model_cfg.get('USE_AUXILIARY_REG', None) == 'point_counts':
            pointcount_map = _sigmoid(self.AUX_PointCount_map(spatial_features_2d))
            pred_dict['pointcount_map'] = pointcount_map.permute(0, 2, 3, 1).contiguous()
        elif self.training and self.model_cfg.get('USE_AUXILIARY_REG', None) == 'corner_cls':
            corner_map = _sigmoid(self.AUX_corner_map(spatial_features_2d))
            pred_dict['corner_map'] = corner_map.permute(0, 2, 3, 1).contiguous()

        for i in range(len(self.box_maps)):
            box_map_name = self.box_map_names[i]
            cur_box_pred = self.box_maps[i](spatial_features_2d)
            if (box_map_name in ['offset']) and (self.model_cfg.TARGET_ASSIGNER_CONFIG.get('HEATMAP_ENCODING_TYPE') is not 'points_count')\
                    and self.model_cfg.LOSS_CONFIG.get('USE_ONENET', False) is False:
                cur_box_pred = _sigmoid(cur_box_pred)
            pred_dict[f'{box_map_name}'] = cur_box_pred.permute(0, 2, 3, 1).contiguous()

        return pred_dict


class CenterHeadMulti(CenterHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class, class_names=class_names,
                         grid_size=grid_size, point_cloud_range=point_cloud_range,
                         predict_boxes_when_training=predict_boxes_when_training, voxel_size=kwargs['voxel_size'])
        self.input_channels = input_channels

        if self.model_cfg.get('SHARED_CONV_NUM_FILTER', None) is not None:
            shared_conv_num_filter = self.model_cfg.get('SHARED_CONV_NUM_FILTER')
            if not isinstance(input_channels, list):
                self.shared_conv = nn.Sequential(
                    nn.Conv2d(input_channels, shared_conv_num_filter, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                    nn.ReLU())
            else:
                self.shared_conv = nn.ModuleList()
                for input_channel in input_channels:
                    curRPN_shared_conv = nn.Sequential(
                        nn.Conv2d(input_channel, shared_conv_num_filter, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                        nn.ReLU())
                    self.shared_conv.append(curRPN_shared_conv)
        else:
            self.shared_conv = None
            shared_conv_num_filter = input_channels

        self.rpn_heads = None
        self.make_multihead(shared_conv_num_filter)

    def make_multihead(self, input_channels):
        rpn_head_cfgs = self.model_cfg.RPN_HEAD_CFGS
        rpn_heads = []
        class_names = []

        if not isinstance(input_channels, list):
            for rpn_head_cfg in rpn_head_cfgs:
                class_names.extend(rpn_head_cfg['HEAD_CLS_NAME'])
                head_label_indices = torch.from_numpy(np.array([
                    self.class_names.index(cur_name) + 1 for cur_name in rpn_head_cfg['HEAD_CLS_NAME']
                ]))
                rpn_head = SingleHead(
                    self.model_cfg, input_channels,
                    len(rpn_head_cfg['HEAD_CLS_NAME']),
                    rpn_head_cfg=rpn_head_cfg, head_label_indices=head_label_indices,
                    separate_reg_config=self.model_cfg.get('SEPARATE_REG_CONFIG', None)
                )
                rpn_heads.append(rpn_head)
        else:
            for rpn_head_cfg, input_channel in zip(rpn_head_cfgs, input_channels):
                class_names.extend(rpn_head_cfg['HEAD_CLS_NAME'])
                head_label_indices = torch.from_numpy(np.array([
                    self.class_names.index(cur_name) + 1 for cur_name in rpn_head_cfg['HEAD_CLS_NAME']
                ]))
                rpn_head = SingleHead(
                    self.model_cfg, input_channel,
                    len(rpn_head_cfg['HEAD_CLS_NAME']),
                    rpn_head_cfg=rpn_head_cfg, head_label_indices=head_label_indices,
                    separate_reg_config=self.model_cfg.get('SEPARATE_REG_CONFIG', None)
                )
                rpn_heads.append(rpn_head)

        self.rpn_heads = nn.ModuleList(rpn_heads)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']  # (B, C, W, H)
        # TODO
        # multi spatial_features_2d || multi input_channels
        if self.shared_conv is not None:
            spatial_features_2d = self.shared_conv(spatial_features_2d)

        ret_dicts = []
        for rpn_head in self.rpn_heads:
            ret_dicts.append(rpn_head(spatial_features_2d))

        # [(B, W, H, class_num), ...]
        # heatmap offset height size orientation
        for key in ret_dicts[0]:
            self.forward_ret_dict[key] = [ret_dict[key] for ret_dict in ret_dicts]

        # 指定目标
        if self.training:
            self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']   # (B, obj_num, 8)
            if 'spatial_points' not in data_dict:
                data_dict['spatial_points'] = None
            targets_dict = self.AssignLabel(
                gt_boxes_classes=data_dict['gt_boxes'], spatial_points=data_dict['spatial_points']
            )
            self.forward_ret_dict.update(targets_dict)

        # 生成box
        if not self.training or self.predict_boxes_when_training:
            pred_hm = [ret_dict['heatmap'].detach() for ret_dict in ret_dicts]
            pred_offset = [ret_dict['offset'].detach() for ret_dict in ret_dicts]
            pred_height = [ret_dict['height'].detach() for ret_dict in ret_dicts]
            pred_size = [ret_dict['size'].detach() for ret_dict in ret_dicts]
            pred_ori = [ret_dict['orientation'].detach() for ret_dict in ret_dicts]
            pred_dicts, recall_dict = self.generate_predicted_boxes(pred_hm=pred_hm, pred_offset=pred_offset,
                                                                    pred_height=pred_height, pred_size=pred_size, pred_ori=pred_ori)
            data_dict['pred_dicts'] = pred_dicts
            data_dict['recall_dict'] = recall_dict
            data_dict['cls_preds_normalized'] = True  # 后处理时不用再sigmoid

            multihead_label_mapping = []
            for idx in range(len(self.rpn_heads)):
                multihead_label_mapping.append(self.rpn_heads[idx].head_label_indices)
            data_dict['multihead_label_mapping'] = multihead_label_mapping

            # new 2021.1.29 , add two stage
            # center points
            centers = []
            boxes = data_dict['pred_dicts']
            num_points = 5
            for box in boxes:
                if num_points == 1 or len(box['pred_boxes']) == 0:
                    centers.append(box['pred_boxes'][:, :3])
                else:
                    height = box['pred_boxes'][:, 2:3]
                    # corners = self.center_to_corner_box2d(center2d, dim2d, rotation_y)
                    corners = box_utils.boxes_to_corners_3d(box['pred_boxes'])
                    corners = corners[:, 0:4, 0:2]

                    front_middle = torch.cat([(corners[:, 0] + corners[:, 1]) / 2, height], dim=-1)
                    back_middle = torch.cat([(corners[:, 2] + corners[:, 3]) / 2, height], dim=-1)
                    left_middle = torch.cat([(corners[:, 0] + corners[:, 3]) / 2, height], dim=-1)
                    right_middle = torch.cat([(corners[:, 1] + corners[:, 2]) / 2, height], dim=-1)

                    # ontnet + 1 keypoint!!
                    if self.model_cfg.LOSS_CONFIG.get('USE_ONENET', False) is True:
                        points = torch.cat([box['pred_boxes'][:, :3], front_middle, back_middle, left_middle, right_middle, box['pred_keypoints']], dim=0)
                    else:
                        points = torch.cat([box['pred_boxes'][:, :3], front_middle, back_middle, left_middle, right_middle], dim=0)
                    centers.append(points)

            data_dict['centers'] = centers

        return data_dict

    def get_hm_losses(self, pred_name, gt_name):
        pred_hms = self.forward_ret_dict[pred_name]  # (B, W, H, class_num)
        gt_hms = self.forward_ret_dict[gt_name]  # (B, class_num, W, H)
        gt_hms = gt_hms.permute(0, 2, 3, 1).contiguous()

        hm_losses = 0
        for idx in range(len(pred_hms)):
            pred_hm = pred_hms[idx]
            head_label_indices = self.rpn_heads[idx].head_label_indices - 1
            gt_hm = gt_hms[..., head_label_indices]

            hm_loss = center_utils.Center_FocalLoss(pred_hm, gt_hm)
            hm_losses += hm_loss
        # 对batch和obj_num的归一化在Center_FocalLoss里
        hm_losses = hm_losses * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS[f'{gt_name}_weight']
        tb_dict = {
            f'{gt_name}_loss': hm_losses.item()
        }
        return hm_losses, tb_dict

    def get_reg_loss(self, reg_name):
        pred_regs = self.forward_ret_dict[reg_name]  # (B, W, H, 2)
        use_BLoss = self.model_cfg.LOSS_CONFIG.get('USE_BalancedL1Loss', False)
        if use_BLoss:
            BLoss_func = center_utils.BalancedL1Loss(alpha=0.5, gamma=1.5)

        if reg_name == 'offset':
            gt_regs = self.forward_ret_dict['anno_box'][:, :, 0:2]  # (B, max_obj, 2)
        elif reg_name == 'height':
            gt_regs = self.forward_ret_dict['anno_box'][:, :, 2:3]  # (B, max_obj, 2)
        elif reg_name == 'size':
            gt_regs = self.forward_ret_dict['anno_box'][:, :, 3:6]  # (B, max_obj, 2)
        elif reg_name == 'orientation':
            gt_regs = self.forward_ret_dict['anno_box'][:, :, 6:8]  # (B, max_obj, 2)
        else:
            raise Exception

        mask = self.forward_ret_dict['mask']  # (batch_size, max_object)
        ind = self.forward_ret_dict['ind']  # (batch_size, max_object)
        cat = self.forward_ret_dict['cat']  # (batch_size, max_object)

        reg_losses = 0
        for idx in range(len(pred_regs)):
            pred_reg = pred_regs[idx]
            pred_reg = pred_reg.view(pred_reg.size(0), -1, pred_reg.size(-1))
            dim = pred_reg.size(-1)

            ind_cp = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
            pred_reg = pred_reg.gather(1, ind_cp)  # !!!!

            head_label_indices = self.rpn_heads[idx].head_label_indices - 1
            cares = torch.zeros_like(mask)
            for label_indice in head_label_indices:
                cared = cat == label_indice
                cares += (cared * 1.0).type_as(mask)
            mask_cares = mask * cares

            if use_BLoss and (reg_name in ['height', 'size']):
                reg_loss = BLoss_func(pred_reg, gt_regs, mask_cares)
            else:
                reg_loss = center_utils.Center_RegLoss(pred_reg, gt_regs, mask_cares)
            reg_losses += reg_loss

        reg_losses = reg_losses * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS[f'{reg_name}_weight']
        tb_dict = {
            reg_name: reg_losses.item()
        }
        return reg_losses, tb_dict

    def get_offset_loss_withRadius(self, radius):
        pred_regs = self.forward_ret_dict['offset']  # (B, W, H, 2)
        gt_regs = self.forward_ret_dict['anno_box'][:, :, 0:2]  # (B, max_obj, 2)

        mask = self.forward_ret_dict['mask']  # (batch_size, max_object)
        ind = self.forward_ret_dict['ind']  # (batch_size, max_object)
        cat = self.forward_ret_dict['cat']  # (batch_size, max_object)

        feater_map_stride = int(self.model_cfg.TARGET_ASSIGNER_CONFIG.get('MAP_STRIDE', 1))
        mapW = int(self.grid_size[1] / feater_map_stride)
        mapH = int(self.grid_size[0] / feater_map_stride)

        gt_regs_withR = gt_regs
        mask_withR = mask
        ind_withR = ind
        cat_withR = cat

        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                ind_cr = ind + x + y*mapH
                second_mask = (ind_cr >= 0) * (ind_cr < mapW*mapH)

                ind_cr = ind_cr * second_mask
                mask_cr = mask * second_mask
                cat_cr = cat * second_mask
                gt_regs_cr = gt_regs + torch.tensor([x, y]).unsqueeze(0).unsqueeze(0).repeat([gt_regs.shape[0], gt_regs.shape[1], 1]).type_as(gt_regs)

                gt_regs_withR = torch.cat([gt_regs_withR, gt_regs_cr], dim=1)
                mask_withR = torch.cat([mask_withR, mask_cr], dim=1)
                ind_withR = torch.cat([ind_withR, ind_cr], dim=1)
                cat_withR = torch.cat([cat_withR, cat_cr], dim=1)

        reg_losses = 0
        for idx in range(len(pred_regs)):
            pred_reg = pred_regs[idx]
            pred_reg = pred_reg.view(pred_reg.size(0), -1, pred_reg.size(-1))
            dim = pred_reg.size(-1)

            ind_cp = ind_withR.unsqueeze(2).expand(ind_withR.size(0), ind_withR.size(1), dim)
            pred_reg = pred_reg.gather(1, ind_cp)  # !!!!

            head_label_indices = self.rpn_heads[idx].head_label_indices - 1
            cares = torch.zeros_like(mask_withR)
            for label_indice in head_label_indices:
                cared = cat_withR == label_indice
                cares += (cared * 1.0).type_as(mask)
            mask_cares = mask_withR * cares

            reg_loss = center_utils.Center_RegLoss(pred_reg, gt_regs_withR, mask_cares)
            reg_losses += reg_loss

        reg_losses = reg_losses * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['offset_weight']
        tb_dict = {
            'offset_withR': reg_losses.item()
        }
        return reg_losses, tb_dict

    # TODO
    # OneNet loss
    def OneNet_Loss(self):
        # 1 match: one to one
        # 2 get loss

        # OneNet config
        focal_loss_alpha = 0.25
        focal_loss_gamma = 2.0

        class_weight = 1.5
        giou_weight = 2.0
        l1_weight = 3.0

        size_weight = 1.0
        ori_weight = 1.0

        def MinCostMatch(pred_batch_box, pred_batch_score, gt_batch_box, gt_batch_cls):
            '''
            :param pred_batch_box:  (H, W, 7)   (x,y,z,dx,dy,dz,ori)
            :param pred_batch_score:    (H, W, cls_num)
            :param gt_batch_box:    (gt_object_num, 7)
            :param gt_batch_cls:    (gt_object_num, 1)
            :return:
             A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            '''
            h, w, k = pred_batch_score.shape
            out_prob = pred_batch_score.reshape(h*w, k)     # (num_queries, num_classes)
            out_bbox = pred_batch_box.reshape(h*w, 7)        # (num_queries, 7)
            gt_batch_cls = gt_batch_cls.reshape(-1).long()

            # Compute the classification cost.
            alpha = focal_loss_alpha
            gamma = focal_loss_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, gt_batch_cls] - neg_cost_class[:, gt_batch_cls]  # output(num_queries, gt_object_num)
            # Compute the L1 cost between boxes
            # cost_bbox = torch.cdist(out_bbox[:, 0:3], gt_batch_box[:, 0:3], p=2)    # output(num_queries, gt_object_num)
            # Compute the giou cost betwen boxes
            # cost_giou = -iou3d_nms_utils.boxes_iou3d_gpu(out_bbox, gt_batch_box)    # output(num_queries, gt_object_num)
            # # Final cost matrix
            # C = l1_weight * cost_bbox + class_weight * cost_class + giou_weight * cost_giou

            # cost_giou = iou3d_nms_utils.boxes_iou3d_gpu(out_bbox, gt_batch_box)  # output(num_queries, gt_object_num)
            # mask_giou = torch.zeros_like(cost_giou)
            # cost_giou = torch.where(cost_giou <= 1.0, cost_giou, mask_giou)
            # cost_giou = 1 - cost_giou

            cost_bbox = torch.cdist(out_bbox[:, 0:2], gt_batch_box[:, 0:2], p=2)  # output(num_queries, gt_object_num)
            # C = l1_weight * cost_bbox + class_weight * cost_class + giou_weight * cost_giou

            cost_height = torch.cdist(out_bbox[:, 2:3], gt_batch_box[:, 2:3], p=1)  # output(num_queries, gt_object_num)
            cost_size_x = torch.cdist(out_bbox[:, 3:4], gt_batch_box[:, 3:4], p=1)  # output(num_queries, gt_object_num)
            cost_size_y = torch.cdist(out_bbox[:, 4:5], gt_batch_box[:, 4:5], p=1)
            cost_size_z = torch.cdist(out_bbox[:, 5:6], gt_batch_box[:, 5:6], p=1)

            cost_ori = torch.cdist(out_bbox[:, 6:7], gt_batch_box[:, 6:7], p=1)
            cost_ori_flip = torch.cdist(out_bbox[:, 6:7], gt_batch_box[:, 6:7] + np.pi, p=1)
            cost_ori = torch.min(cost_ori, cost_ori_flip)

            # Final cost matrix
            C = l1_weight * cost_bbox + class_weight * cost_class + size_weight*(cost_size_x + cost_size_y + cost_size_z + cost_height) + ori_weight*cost_ori

            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(len(gt_batch_cls)).to(src_ind)
            return (src_ind, tgt_ind)

        def _get_src_permutation_idx(indices):
            # permute predictions following indices
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _) in indices])
            return batch_idx, src_idx

        def loss_labels(pred_score, targets, indices, num_boxes, log=False):
            """Classification loss (NLL)
            pred_score:    (B ,H, W, cls_num)
            targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
            """
            bs, h, w, k = pred_score.shape
            src_logits = pred_score.reshape(bs, h * w, k)

            idx = _get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], k, dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            src_logits = src_logits.flatten(0, 1)
            # prepare one_hot target.
            target_classes = target_classes.flatten(0, 1)
            pos_inds = torch.nonzero(target_classes != k, as_tuple=True)[0]
            labels = torch.zeros_like(src_logits)
            labels[pos_inds, target_classes[pos_inds]] = 1
            # comp focal loss.
            class_loss = onenet_utils.onenet_focal_loss_jit(
                src_logits,
                labels,
                alpha=focal_loss_alpha,
                gamma=focal_loss_gamma,
                reduction="sum",
            ) / num_boxes

            return class_loss

        def loss_boxes(pred_boxes, targets, indices, num_boxes):
            """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
               targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
               The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
            """
            idx = _get_src_permutation_idx(indices)

            bs, h, w, k = pred_boxes.shape
            src_boxes = pred_boxes.reshape(bs, h * w, k)

            src_boxes = src_boxes[idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            loss_giou = 1 - torch.diag(iou3d_nms_utils.boxes_iou3d_gpu(src_boxes, target_boxes))
            loss_giou = loss_giou.sum() / num_boxes

            loss_bbox = F.mse_loss(src_boxes[..., 0:2], target_boxes[..., 0:2], reduction='none')
            loss_bbox = loss_bbox.sum() / num_boxes

            return loss_giou, loss_bbox

        def loss_boxes_v2(pred_boxes, targets, indices, num_boxes):
            """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
               targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
               The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
            """
            idx = _get_src_permutation_idx(indices)

            bs, h, w, k = pred_boxes.shape
            src_boxes = pred_boxes.reshape(bs, h * w, k)

            src_boxes = src_boxes[idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            loss_bbox = F.mse_loss(src_boxes[..., 0:3], target_boxes[..., 0:3], reduction='none')
            loss_bbox = loss_bbox.sum() / num_boxes

            loss_size_x = F.l1_loss(src_boxes[..., 3:4], target_boxes[..., 3:4], reduction='none')
            loss_size_y = F.l1_loss(src_boxes[..., 4:5], target_boxes[..., 4:5], reduction='none')
            loss_size_z = F.l1_loss(src_boxes[..., 5:6], target_boxes[..., 5:6], reduction='none')
            loss_ori = F.l1_loss(src_boxes[..., 6:7], target_boxes[..., 6:7], reduction='none')

            loss_size = (loss_size_x + loss_size_y + loss_size_z).sum() / num_boxes
            loss_ori = loss_ori.sum() / num_boxes

            return loss_bbox, loss_size, loss_ori

        def loss_boxes_v3(pred_boxes, targets, indices, num_boxes):
            """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
               targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
               The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
            """
            idx = _get_src_permutation_idx(indices)

            bs, h, w, k = pred_boxes.shape
            src_boxes = pred_boxes.reshape(bs, h * w, k)

            src_boxes = src_boxes[idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            loss_bbox = F.mse_loss(src_boxes[..., 0:2], target_boxes[..., 0:2], reduction='none')
            loss_bbox = loss_bbox.sum() / num_boxes

            loss_bbox_height = F.l1_loss(src_boxes[..., 2:3], target_boxes[..., 2:3], reduction='none')
            loss_size_x = F.l1_loss(src_boxes[..., 3:4], target_boxes[..., 3:4], reduction='none')
            loss_size_y = F.l1_loss(src_boxes[..., 4:5], target_boxes[..., 4:5], reduction='none')
            loss_size_z = F.l1_loss(src_boxes[..., 5:6], target_boxes[..., 5:6], reduction='none')

            loss_ori = F.l1_loss(src_boxes[..., 6:7], target_boxes[..., 6:7], reduction='none')
            loss_ori_flip = F.l1_loss(src_boxes[..., 6:7], target_boxes[..., 6:7] + np.pi, reduction='none')
            loss_ori = torch.min(loss_ori, loss_ori_flip)

            loss_size = (loss_size_x + loss_size_y + loss_size_z + loss_bbox_height).sum() / num_boxes
            loss_ori = loss_ori.sum() / num_boxes

            return loss_bbox, loss_size, loss_ori

        def loss_boxes_v4(pred_boxes, targets, indices, num_boxes):
            """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
               targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
               The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
            """
            idx = _get_src_permutation_idx(indices)

            bs, h, w, k = pred_boxes.shape
            src_boxes = pred_boxes.reshape(bs, h * w, k)

            src_boxes = src_boxes[idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            loss_bbox = F.mse_loss(src_boxes[..., 0:2], target_boxes[..., 0:2], reduction='none')
            loss_bbox = loss_bbox.sum() / num_boxes

            loss_giou = loss_utils.get_corner_loss_lidar(src_boxes.reshape(-1, k), target_boxes.reshape(-1, k))
            loss_giou = loss_giou.sum() / num_boxes

            return loss_giou, loss_bbox

        feater_map_stride = int(self.model_cfg.TARGET_ASSIGNER_CONFIG.get('MAP_STRIDE', 1))
        tb_dict = {
            'loss_class': 0,
            'loss_giou': 0,
            'loss_bbox': 0
        }
        total_losses = 0

        gt_boxes_classes = self.forward_ret_dict['gt_boxes']  # (B, max_object, 8)
        gt_boxes = gt_boxes_classes[:, :, :-1]
        gt_classes = gt_boxes_classes[:, :, -1] - 1
        batch_size, max_object, obj_encode_num = gt_boxes.shape
        mask = self.forward_ret_dict['mask']  # (batch_size, max_object)
        ind = self.forward_ret_dict['ind']  # (batch_size, max_object)
        cat = self.forward_ret_dict['cat']  # (batch_size, max_object)

        pred_hms = self.forward_ret_dict['heatmap']  # (B, W, H, class_num)
        pred_offsets = self.forward_ret_dict['offset']  # (B, W, H, 2)
        pred_heights = self.forward_ret_dict['height']  # (B, W, H, 2)
        pred_sizes = self.forward_ret_dict['size']  # (B, W, H, 2)
        pred_orientations = self.forward_ret_dict['orientation']  # (B, W, H, 2)

        for idx in range(len(pred_hms)):
            pred_hm = pred_hms[idx]
            pred_offset = pred_offsets[idx]
            pred_height = pred_heights[idx]
            pred_size = pred_sizes[idx]
            pred_orientation = pred_orientations[idx]
            batch_size, H, W, num_cls = pred_hm.size()

            # pred_height = pred_height.relu_()
            # pred_size = pred_size.relu_()

            batch_rot = torch.atan2(pred_orientation[..., 0:1], pred_orientation[..., 1:2])
            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W, 1).repeat(batch_size, 1, 1, 1).to(pred_hm.device).float()
            xs = xs.view(1, H, W, 1).repeat(batch_size, 1, 1, 1).to(pred_hm.device).float()

            xs = xs + pred_offset[..., 0:1]
            ys = ys + pred_offset[..., 1:2]
            xs = xs * self.voxel_size[0] * feater_map_stride + self.point_cloud_range[0]
            ys = ys * self.voxel_size[1] * feater_map_stride + self.point_cloud_range[1]
            pred_boxes = torch.cat([xs, ys, pred_height, pred_size, batch_rot], dim=3)

            head_label_indices = self.rpn_heads[idx].head_label_indices - 1
            cares = torch.zeros_like(mask)
            cat_projected = cat.new_zeros(cat.shape)
            tt = 0
            for label_indice in head_label_indices:
                cared = cat == label_indice
                cares += (cared * 1.0).type_as(mask)
                cat_projected[cared] = tt
                tt += 1
            mask_cares = mask * cares
            mask_cares = mask_cares.bool()

            indices = []
            targets = []
            for bat in range(batch_size):
                mask_batch_box = gt_boxes[bat, mask_cares[bat], :].reshape(-1, 7)
                mask_batch_box_cls = cat_projected[bat, mask_cares[bat]].reshape(-1)
                if mask_batch_box_cls.shape[0] == 0:
                    indices.append(([], []))
                    continue
                batch_targets = {
                    'labels': mask_batch_box_cls.long(),
                    'boxes': mask_batch_box
                }
                targets.append(batch_targets)
                # batch_rot = torch.atan2(pred_orientation[bat, :, :, 0:1], pred_orientation[bat, :, :, 1:2])
                # ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
                # ys = ys.view(H, W).to(pred_hm.device).float()
                # xs = xs.view(H, W).to(pred_hm.device).float()
                #
                # xs = xs + pred_offset[bat, :, :, 0]
                # ys = ys + pred_offset[bat, :, :, 1]
                # xs = xs * self.voxel_size[0] * feater_map_stride + self.point_cloud_range[0]
                # ys = ys * self.voxel_size[1] * feater_map_stride + self.point_cloud_range[1]
                # pred_batch_box = torch.cat([xs, ys, pred_height[bat], pred_size[bat], batch_rot], dim=2)
                pred_batch_box = pred_boxes[bat]

                bat_indices = MinCostMatch(pred_batch_box, pred_hm[bat], mask_batch_box, mask_batch_box_cls)
                indices.append(bat_indices)
            match_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            num_boxes = sum(len(t['labels']) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_hm.device)
            num_boxes = torch.clamp(num_boxes, min=1).item()

            class_loss = loss_labels(pred_score=pred_hm, targets=targets, indices=match_indices, num_boxes=num_boxes)

            # loss_giou, loss_bbox = loss_boxes_v4(pred_boxes=pred_boxes, targets=targets, indices=match_indices, num_boxes=num_boxes)
            #
            # class_loss = class_loss*class_weight
            # loss_giou = loss_giou*giou_weight
            # loss_bbox = loss_bbox*l1_weight
            # total_losses = total_losses + class_loss + loss_giou + loss_bbox

            loss_bbox, loss_size, loss_ori = loss_boxes_v3(pred_boxes=pred_boxes, targets=targets, indices=match_indices, num_boxes=num_boxes)
            class_loss = class_loss * class_weight
            loss_giou = loss_size * loss_size + loss_ori * loss_ori
            loss_bbox = loss_bbox * l1_weight
            total_losses = total_losses + class_loss + loss_giou + loss_bbox

            tb_dict['loss_class'] += class_loss.item()
            tb_dict['loss_giou'] += loss_giou.item()
            tb_dict['loss_bbox'] += loss_bbox.item()
        return total_losses, tb_dict

    def get_loss(self):
        if self.model_cfg.LOSS_CONFIG.get('USE_ONENET', False) is True:
            rpn_loss, tb_dict = self.OneNet_Loss()
        else:
            # heatmap offset height size orientation
            hm_loss, tb_dict = self.get_hm_losses(pred_name='heatmap', gt_name='hm')

            reg_losses = 0
            for reg_name in self.rpn_heads[0].box_map_names:
                if reg_name == 'offset' and self.model_cfg.TARGET_ASSIGNER_CONFIG.get('OFFSET_RADIUS', 0) != 0:
                    radius = int(self.model_cfg.TARGET_ASSIGNER_CONFIG.get('OFFSET_RADIUS', 0))
                    reg_loss, tb_dict_offset = self.get_offset_loss_withRadius(radius)
                else:
                    reg_loss, tb_dict_offset = self.get_reg_loss(reg_name)
                tb_dict.update(tb_dict_offset)
                reg_losses += reg_loss

            rpn_loss = hm_loss + reg_losses

        if self.training and self.model_cfg.get('USE_AUXILIARY_REG', None) == 'point_counts':
            pc_loss, tb_dict_offset = self.get_hm_losses(pred_name='pointcount_map', gt_name='point_counts')
            tb_dict.update(tb_dict_offset)
            rpn_loss += pc_loss
        elif self.training and self.model_cfg.get('USE_AUXILIARY_REG', None) == 'corner_cls':
            pc_loss, tb_dict_offset = self.get_hm_losses(pred_name='corner_map', gt_name='corner_cls')
            tb_dict.update(tb_dict_offset)
            rpn_loss += pc_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict