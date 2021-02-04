import torch
import torch.nn as nn
import numpy as np
from .center_head_template import CenterHeadTemplate, _sigmoid
from ...utils import weight_init_utils


class CenterHeadSingle(CenterHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class, class_names=class_names,
                         grid_size=grid_size, point_cloud_range=point_cloud_range,
                         predict_boxes_when_training=predict_boxes_when_training, voxel_size=kwargs['voxel_size'])

        # *******************************************************************
        # **************************** CenterPoint *********************************
        # *******************************************************************

        # self.shared_conv = nn.Sequential(
        #     nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=True),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )
        # input_channels = 64

        # self.shared_conv = nn.Sequential(
        # nn.ConvTranspose2d(
        #     input_channels, 128, 2,
        #     stride=2, bias=False
        # ),
        # nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        # nn.ReLU()
        # )
        # input_channels = 128
        #
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        input_channels = 64
        head_conv = 64
        head_kernel = 3
        final_kernel = 1

        self.heat_map = nn.Sequential(
            nn.Conv2d(input_channels, head_conv, kernel_size=head_kernel, stride=1, padding=head_kernel // 2, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(),
            nn.Conv2d(head_conv, num_class, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
        )
        self.offset_map = nn.Sequential(
            nn.Conv2d(input_channels, head_conv, kernel_size=head_kernel, stride=1, padding=head_kernel // 2, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(),
            nn.Conv2d(head_conv, 2, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
        )
        self.height_map = nn.Sequential(
            nn.Conv2d(input_channels, head_conv, kernel_size=head_kernel, stride=1, padding=head_kernel // 2, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(),
            nn.Conv2d(head_conv, 1, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
        )
        self.size_map = nn.Sequential(
            nn.Conv2d(input_channels, head_conv, kernel_size=head_kernel, stride=1, padding=head_kernel // 2, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(),
            nn.Conv2d(head_conv, 3, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
        )
        # TODO
        # afdet paper's 8 orientation
        self.orientation_map = nn.Sequential(
            nn.Conv2d(input_channels, head_conv, kernel_size=head_kernel, stride=1, padding=head_kernel // 2, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(),
            nn.Conv2d(head_conv, 2, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
        )

        # 影响大吗？
        self.init_weight()

    def init_weight(self):
        # init as center_point
        # 这里对hm头的初始化！！！
        init_bias = -2.19
        self.heat_map[-1].bias.data.fill_(init_bias)

        for m in self.offset_map.modules():
            if isinstance(m, nn.Conv2d):
                weight_init_utils.kaiming_init(m)

        for m in self.height_map.modules():
            if isinstance(m, nn.Conv2d):
                weight_init_utils.kaiming_init(m)

        for m in self.size_map.modules():
            if isinstance(m, nn.Conv2d):
                weight_init_utils.kaiming_init(m)

        for m in self.orientation_map.modules():
            if isinstance(m, nn.Conv2d):
                weight_init_utils.kaiming_init(m)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']  #(B, C, W, H)
        spatial_features_2d = self.shared_conv(spatial_features_2d)

        heatmap_pred = _sigmoid(self.heat_map(spatial_features_2d))   #(B, class_num, W, H)
        offset_pred = _sigmoid(self.offset_map(spatial_features_2d))  #(B, 2, W, H)
        # heatmap_pred = self.heat_map(spatial_features_2d)  # (B, class_num, W, H)
        # offset_pred = self.offset_map(spatial_features_2d)  # (B, 2, W, H)
        height_pred = self.height_map(spatial_features_2d)  #(B, 1, W, H)
        size_pred = self.size_map(spatial_features_2d)      #(B, 3, W, H)
        orientation_pred = self.orientation_map(spatial_features_2d)   #(B, 8, W, H)

        # (B, W, H, class_num)
        self.forward_ret_dict['heatmap'] = heatmap_pred.permute(0, 2, 3, 1).contiguous()
        self.forward_ret_dict['offset'] = offset_pred.permute(0, 2, 3, 1).contiguous()
        self.forward_ret_dict['height'] = height_pred.permute(0, 2, 3, 1).contiguous()
        self.forward_ret_dict['size'] = size_pred.permute(0, 2, 3, 1).contiguous()
        self.forward_ret_dict['orientation'] = orientation_pred.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']   # (B, obj_num, 8)
        # 指定目标
        if self.training:
            targets_dict = self.AssignLabel(
                gt_boxes_classes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        # 生成box
        if not self.training or self.predict_boxes_when_training:
            pred_dicts, recall_dict = self.generate_predicted_boxes()
            data_dict['pred_dicts'] = pred_dicts
            data_dict['recall_dict'] = recall_dict
            data_dict['cls_preds_normalized'] = True        # 后处理时不用再sigmoid

        return data_dict
