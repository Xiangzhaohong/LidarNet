import numpy as np
import torch
import torch.nn as nn
from ...utils import weight_init_utils


# Attentionally Spatial-semantic RPN
class SSFA(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super(SSFA, self).__init__()
        self.model_cfg = model_cfg
        norm_cfg = None
        name = "rpn"

        self._layer_strides = self.model_cfg.get('ds_layer_strides', None)  # [1,]
        self._num_filters = self.model_cfg.get('ds_num_filters', None)      # [128,]
        self._layer_nums = self.model_cfg.get('layer_nums', None)           # [5,]
        self._upsample_strides = self.model_cfg.get('us_layer_strides', None)      # [1,]
        self._num_upsample_filters = self.model_cfg.get('us_num_filters', None)    # [128,]
        self._num_input_features = input_channels  # 128
        print(input_channels)

        if norm_cfg is None:  # True
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        self.bottom_up_block_0 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(self._num_input_features, self._num_input_features, 3, stride=1, bias=False),
            # build_norm_layer(self._norm_cfg, 128)[1],
            nn.BatchNorm2d(self._num_input_features, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(in_channels=self._num_input_features, out_channels=self._num_input_features, kernel_size=3, stride=1, padding=1, bias=False, ),
            # build_norm_layer(self._norm_cfg, 128, )[1],
            nn.BatchNorm2d(self._num_input_features, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(in_channels=self._num_input_features, out_channels=self._num_input_features, kernel_size=3, stride=1, padding=1, bias=False, ),
            # build_norm_layer(self._norm_cfg, 128, )[1],
            nn.BatchNorm2d(self._num_input_features, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.bottom_up_block_1 = nn.Sequential(
            # [200, 176] -> [100, 88]
            nn.Conv2d(in_channels=self._num_input_features, out_channels=self._num_input_features * 2, kernel_size=3, stride=2, padding=1, bias=False, ),
            # build_norm_layer(self._norm_cfg, 256, )[1],
            nn.BatchNorm2d(self._num_input_features * 2, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(in_channels=self._num_input_features * 2, out_channels=self._num_input_features * 2, kernel_size=3, stride=1, padding=1, bias=False, ),
            # build_norm_layer(self._norm_cfg, 256, )[1],
            nn.BatchNorm2d(self._num_input_features * 2, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(in_channels=self._num_input_features * 2, out_channels=self._num_input_features * 2, kernel_size=3, stride=1, padding=1, bias=False, ),
            # build_norm_layer(self._norm_cfg, 256, )[1],
            nn.BatchNorm2d(self._num_input_features * 2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.trans_0 = nn.Sequential(
            nn.Conv2d(in_channels=self._num_input_features, out_channels=self._num_input_features, kernel_size=1, stride=1, padding=0, bias=False, ),
            # build_norm_layer(self._norm_cfg, 128, )[1],
            nn.BatchNorm2d(self._num_input_features, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.trans_1 = nn.Sequential(
            nn.Conv2d(in_channels=self._num_input_features * 2, out_channels=self._num_input_features * 2, kernel_size=1, stride=1, padding=0, bias=False, ),
            # build_norm_layer(self._norm_cfg, 256, )[1],
            nn.BatchNorm2d(self._num_input_features * 2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.deconv_block_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self._num_input_features * 2, out_channels=self._num_input_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            # build_norm_layer(self._norm_cfg, 128, )[1],
            nn.BatchNorm2d(self._num_input_features, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self._num_input_features * 2, out_channels=self._num_input_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            # build_norm_layer(self._norm_cfg, 128, )[1],
            nn.BatchNorm2d(self._num_input_features, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=self._num_input_features, out_channels=self._num_input_features, kernel_size=3, stride=1, padding=1, bias=False, ),
            # build_norm_layer(self._norm_cfg, 128, )[1],
            nn.BatchNorm2d(self._num_input_features, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.w_0 = nn.Sequential(
            nn.Conv2d(in_channels=self._num_input_features, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            # build_norm_layer(self._norm_cfg, 1, )[1],
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=self._num_input_features, out_channels=self._num_input_features, kernel_size=3, stride=1, padding=1, bias=False, ),
            # build_norm_layer(self._norm_cfg, 128, )[1],
            nn.BatchNorm2d(self._num_input_features, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.w_1 = nn.Sequential(
            nn.Conv2d(in_channels=self._num_input_features, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            # build_norm_layer(self._norm_cfg, 1, )[1],
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )
        self.num_bev_features = self._num_input_features
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init_utils.xavier_init(m, distribution="uniform")

    def forward(self, data_dict):
        x = data_dict['spatial_features']
        x_0 = self.bottom_up_block_0(x)
        x_1 = self.bottom_up_block_1(x_0)
        x_trans_0 = self.trans_0(x_0)
        x_trans_1 = self.trans_1(x_1)
        x_middle_0 = self.deconv_block_0(x_trans_1) + x_trans_0
        x_middle_1 = self.deconv_block_1(x_trans_1)
        x_output_0 = self.conv_0(x_middle_0)
        x_output_1 = self.conv_1(x_middle_1)

        x_weight_0 = self.w_0(x_output_0)
        x_weight_1 = self.w_1(x_output_1)
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1)
        x_output = x_output_0 * x_weight[:, 0:1, :, :] + x_output_1 * x_weight[:, 1:, :, :]

        data_dict['spatial_features_2d'] = x_output.contiguous()

        return data_dict
