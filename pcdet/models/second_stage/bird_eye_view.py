import numpy as np
import torch
import torch.nn as nn


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class BEVFeatureExtractor(nn.Module): 
    def __init__(self, model_cfg, pc_start, voxel_size, out_stride, num_point):
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_start = pc_start 
        self.voxel_size = voxel_size
        self.out_stride = out_stride
        self.num_point = num_point

    def absl_to_relative(self, absolute):
        a1 = (absolute[..., 0] - self.pc_start[0]) / self.voxel_size[0] / self.out_stride 
        a2 = (absolute[..., 1] - self.pc_start[1]) / self.voxel_size[1] / self.out_stride 

        return a1, a2

    def forward(self, data_dict):
        batch_centers = data_dict['centers']
        batch_size = len(data_dict['spatial_features_2d'])  #(B, C, W, H)
        features = data_dict['spatial_features_2d'].permute(0, 2, 3, 1).contiguous()
        ret_maps = []
        # print(batch_size)

        for batch_idx in range(batch_size):
            xs, ys = self.absl_to_relative(batch_centers[batch_idx])
            # N x C 
            feature_map = bilinear_interpolate_torch(features[batch_idx], xs, ys)

            if self.num_point > 1:
                section_size = len(feature_map) // self.num_point
                feature_map = torch.cat([feature_map[i*section_size: (i+1)*section_size] for i in range(self.num_point)], dim=1)

            ret_maps.append(feature_map)

        data_dict['ret_maps'] = ret_maps

        return data_dict
