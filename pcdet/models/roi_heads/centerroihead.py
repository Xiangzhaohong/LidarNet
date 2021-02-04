import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from ..model_utils.model_nms_utils import class_agnostic_nms


class CenterROIHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, code_size=7):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        pre_channel = input_channels

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def reorder_first_stage_pred_and_feature(self, batch_dict, nms_config):
        first_pred = batch_dict['pred_dicts']
        features = batch_dict['ret_maps']
        batch_size = len(first_pred)
        box_length = first_pred[0]['pred_boxes'].shape[1]
        feature_vector_length = features[0].shape[-1]

        rois = first_pred[0]['pred_boxes'].new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, box_length))
        roi_scores = first_pred[0]['pred_scores'].new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = first_pred[0]['pred_labels'].new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)
        roi_features = features[0].new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, feature_vector_length))

        for i in range(batch_size):
            num_obj = features[i].shape[0]
            # print(num_obj)
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target
            box_preds = first_pred[i]['pred_boxes']
            cls_preds = first_pred[i]['pred_labels']
            scores_preds = first_pred[i]['pred_scores']

            # if nms_config.MULTI_CLASSES_NMS:
            #     raise NotImplementedError
            # else:
            #     selected, selected_scores = class_agnostic_nms(
            #         box_scores=scores_preds, box_preds=box_preds, nms_config=nms_config
            #     )
            # print(selected)
            # print(box_preds.shape)

            rois[i, :num_obj] = box_preds
            roi_labels[i, :num_obj] = cls_preds
            roi_scores[i, :num_obj] = scores_preds
            roi_features[i, :num_obj] = features[i]

        batch_dict['rois'] = rois
        batch_dict['roi_labels'] = roi_labels
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_features'] = roi_features
        batch_dict['has_class_labels'] = True

        return batch_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = forward_ret_dict['rcnn_reg'].shape[-1]
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'L1':
            reg_targets = gt_boxes3d_ct.view(rcnn_batch_size, -1)
            rcnn_loss_reg = F.l1_loss(
                rcnn_reg.view(rcnn_batch_size, -1),
                reg_targets,
                reduction='none'
            )  # [B, M, 7]

            rcnn_loss_reg = rcnn_loss_reg * rcnn_loss_reg.new_tensor(loss_cfgs.LOSS_WEIGHTS['code_weights'])

            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(
                fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = box_preds.shape[-1]
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)

        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = (batch_box_preds + local_rois).view(-1, code_size)
        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)

        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)

        return batch_cls_preds, batch_box_preds

    def assign_targets(self, batch_dict):
        import numpy as np

        def limit_period(val, offset=0.5, period=np.pi):
            return val - torch.floor(val / period + offset) * period

        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        roi_ry = limit_period(rois[:, :, 6], offset=0.5, period=np.pi * 2)

        gt_of_rois[:, :, :6] = gt_of_rois[:, :, :6] - rois[:, :, :6]
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        batch_dict = self.reorder_first_stage_pred_and_feature(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_features'] = targets_dict['roi_features']

        # RoI aware pooling
        pooled_features = batch_dict['roi_features'].reshape(-1, 1, batch_dict['roi_features'].shape[-1]).contiguous()  # (BxN, 1, C)

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).contiguous()  # (BxN, C, 1)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
