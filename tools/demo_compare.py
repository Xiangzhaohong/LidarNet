import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils, box_utils
from visual_utils import visualize_utils as V


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str,
                        default='/home/syang/Data/data_object_velodyne/output/kitti_models/onenet_twostage_0130/test/onenet_twostage_0130.yaml',
                        help='specify the config for training')
    parser.add_argument('--ckpt', type=str,
                        default='/home/syang/Data/data_object_velodyne/output/kitti_models/onenet_twostage_0130/test/ckpt/checkpoint_epoch_78.pth',
                        help='checkpoint to start from')
    parser.add_argument('--show_heatmap', action='store_true', default=False, help='')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    return args, cfg


def main():
    args, cfg = parse_config()
    cfg.ROOT_DIR = Path(cfg.DATA_CONFIG.DATA_PATH)
    logger = common_utils.create_logger()
    dist_test = False
    total_gpus = 1

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, batch_dict in enumerate(test_loader):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)

            filtered_gt_boxes = batch_dict['gt_boxes'][0].cpu().numpy()

            mask = box_utils.mask_boxes_outside_range_numpy(filtered_gt_boxes, test_loader.dataset.point_cloud_range)
            filtered_gt_boxes = filtered_gt_boxes[mask]

            if args.show_heatmap:
                pass
            if 'pred_keypoints' in pred_dicts[0]:
                pred_keypoints = pred_dicts[0]['pred_keypoints']
            else:
                pred_keypoints = None
            V.draw_scenes(
                points=batch_dict['points'][:, 1:],
                gt_boxes=filtered_gt_boxes[:, :-1], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                gt_labels=filtered_gt_boxes[:, -1],
                class_names=test_loader.dataset.class_names,
                pred_keypoints=pred_keypoints
            )
            mlab.show(stop=True)

    logger.info('Demo done.')


def draw_heatmap(heatmap, boxes):
    pass

if __name__ == '__main__':
    main()

