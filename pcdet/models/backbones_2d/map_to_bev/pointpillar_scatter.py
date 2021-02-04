import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords, num_points = batch_dict['pillar_features'], batch_dict['voxel_coords'], batch_dict['voxel_num_points']
        batch_spatial_features = []
        batch_spatial_points = []

        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            spatial_points = torch.zeros(
                1,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

            num_point = num_points[batch_mask]
            spatial_points[:, indices] = num_point
            batch_spatial_points.append(spatial_points)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_points = torch.stack(batch_spatial_points, 0)

        # x 和 y 相反的罪魁祸首
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_spatial_points = batch_spatial_points.view(batch_size, 1 * self.nz, self.ny, self.nx)
        batch_dict['spatial_points'] = batch_spatial_points
        return batch_dict
