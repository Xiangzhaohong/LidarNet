import copy
import pickle

import numpy as np

from pcdet.utils import box_utils, common_utils
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.datasets.robosense import robosense_utils


class RobosenseDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.include_robosense_data(self.mode)
        # balanced_infos_resampling as nuscenes
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

    def include_robosense_data(self, mode):
        self.logger.info('Loading RoboSense dataset')
        robosense_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                self.logger.info('Info_Path not exist in mode %s' % mode)
                continue
            else:
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    robosense_infos.extend(infos)

        for i in range(len(robosense_infos)):
            robosense_infos[i]['annos']['gt_names'] = np.array([robosense_utils.map_name_from_general_to_detection[name]
                                                                for name in robosense_infos[i]['annos']['gt_names']])
        self.infos.extend(robosense_infos)
        self.logger.info('Total samples for robosense dataset: %d, for mode: %s' % (len(robosense_infos), mode))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            # 每一帧为复制单位, 是有重复帧的
            for name in set(info['annos']['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v)/duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []
        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, cur_cls_ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * cur_cls_ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        return sampled_infos

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def __getitem__(self, index):

        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = robosense_utils.load_pcd_to_ndarray(info['lidar_path'])
        # 反射率归一化
        points[:, 3] = points[:, 3] / 255.0
        annos = info['annos']
        input_dict = {
            'points': points,
            'frame_id': annos['frame_id']
        }

        if 'gt_boxes_lidar' in annos:
            # 在这去掉don't care/unknown吗?
            annos = robosense_utils.drop_info_with_name(annos, name='unknown') #name='DontCare')

            # # 去掉box里面点数很少的box    是否有用?
            # annos = robosense_utils.drop_info_with_box_points(annos, min_pts=5)

            # 是否有用?
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (annos['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            gt_names = annos['gt_names'] if mask is None else annos['gt_names'][mask]
            gt_boxes_lidar = annos['gt_boxes_lidar'] if mask is None else annos['gt_boxes_lidar'][mask]

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
                Args:
                    batch_dict:
                        frame_id:
                    pred_dicts: list of pred_dicts
                        pred_boxes: (N, 7), Tensor
                        pred_scores: (N), Tensor
                        pred_labels: (N), Tensor
                    class_names:
                    output_path:
                Returns:
                """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            # 这个确实可以这样用...
            pred_dict['name'] = np.array(class_names)[pred_labels.astype(np.int) - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]

            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        eval_det_annos = copy.deepcopy(det_annos)   # 'frame_id' 'name' 'score' 'boxes_lidar' 'pred_labels'
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]   # 'gt_names' 'gt_boxes_lidar' 'num_lidar_pts' 'frame_id'
        for i in range(len(eval_gt_annos)):
            eval_gt_annos[i]['name'] = eval_gt_annos[i]['gt_names']

        # for test
        eval_gt_annos = eval_gt_annos[0:len(eval_det_annos)]
        assert len(eval_gt_annos) == len(eval_det_annos)
        for i in range(len(eval_det_annos)):
            assert eval_det_annos[i]['frame_id'] == eval_gt_annos[i]['frame_id']

        # Filter boxes (distance, points per box, etc.).
        filtered_gt_box_num_By_range = 0
        for i in range(len(eval_gt_annos)):
            mask = box_utils.mask_boxes_outside_range_numpy(eval_gt_annos[i]['gt_boxes_lidar'], self.point_cloud_range)
            filtered_gt_box_num_By_range += (eval_gt_annos[i]['gt_boxes_lidar'].shape[0] - mask.sum())
            for key in eval_gt_annos[i].keys():
                if isinstance(eval_gt_annos[i][key], np.ndarray):
                    eval_gt_annos[i][key] = eval_gt_annos[i][key][mask]
        self.logger.info('Eval preprocess--filter gt box by PointCloud range. filtered %d GT boxes' % (int(filtered_gt_box_num_By_range)))

        # # filtered_dt_box_num_By_score
        # min_score = 0.2
        # for i in range(len(eval_det_annos)):
        #     keep_indices = [i for i, x in enumerate(eval_det_annos[i]['score']) if x >= min_score]
        #     for key in eval_det_annos[i].keys():
        #         if isinstance(eval_det_annos[i][key], np.ndarray):
        #             eval_det_annos[i][key] = eval_det_annos[i][key][keep_indices]

        # filtered_gt_box_num_By_pointsnum = 0
        # min_pts = 5
        # for i in range(len(eval_gt_annos)):
        #     mask = eval_gt_annos[i]['boxes_points_pts'] >= min_pts
        #     filtered_gt_box_num_By_pointsnum += (eval_gt_annos[i]['gt_boxes_lidar'].shape[0] - mask.sum())
        #     for key in eval_gt_annos[i].keys():
        #         eval_gt_annos[i][key] = eval_gt_annos[i][key][mask]
        # self.logger.info('Eval preprocess--filter gt box by box points num. filtered %d GT boxes' % (
        #     int(filtered_gt_box_num_By_pointsnum)))

        #
        print(class_names)
        total_num_class_det = np.zeros(len(class_names), dtype=np.int)
        total_num_class_gt = np.zeros(len(class_names), dtype=np.int)

        for i in range(len(eval_det_annos)):
            name_ints = np.array([class_names.index(n) for n in eval_det_annos[i]['name']], dtype=np.int32)
            for name_int in name_ints:
                total_num_class_det[name_int] += 1

            name_ints = np.array([class_names.index(n) for n in eval_gt_annos[i]['name'] if n in class_names], dtype=np.int32)
            for name_int in name_ints:
                total_num_class_gt[name_int] += 1
        print('Det total num class: ', total_num_class_det.tolist())
        print('GT total num class: ', total_num_class_gt.tolist())
        # TODO
        # 是否在这里去掉unknown/dontcare类 ||| 把未检测的类都映射成dont care
        # common_utils.drop_info_with_name

        def robosense_eval():
            from . import robosense_eval
            # some parameter needed here
            using_IOU = True
            overlap_0_7 = np.array([0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5])
            min_overlaps = overlap_0_7
            # TODO
            # now just part for robosense
            # class_to_name = {
            #     0: 'vehicle',
            #     1: 'pedestrian',
            #     2: 'bicycle',
            #
            # }
            class_to_name = {
                0: 'vehicle',
                1: 'tricycle',
                2: 'big_vehicle',
                3: 'huge_vehicle',
                4: 'motorcycle',
                5: 'bicycle',
                6: 'pedestrian',
                7: 'cone'
            }

            my_eval = robosense_eval.RoboSense_Eval(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos,
                current_classes=class_names, output_dir=self.root_path,
                class_to_name=class_to_name, min_overlaps=min_overlaps
            )
            ap_result_str, ap_dict = my_eval.evaluate()
            return ap_result_str, ap_dict

        # 直接转到kitti下面,用kitti的标准评估
        def kitti_eval():
            from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
            from pcdet.datasets.kitti import kitti_utils

            map_name_to_kitti = {
                'vehicle': 'Car',
                'tricycle': 'Tricycle',
                'big_vehicle': 'Big_vehicle',
                'huge_vehicle': 'Huge_vehicle',
                'motorcycle': 'Motorcycle',
                'bicycle': 'Cyclist',
                'pedestrian': 'Pedestrian',
                'cone': 'Cone',
                'unknown': 'DontCare'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(eval_gt_annos, map_name_to_kitti=map_name_to_kitti)

            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            PR_detail_dict = {}
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names,
                PR_detail_dict=PR_detail_dict
            )
            return ap_result_str, ap_dict

        ap_result_str, ap_dict = kitti_eval() #robosense_eval()
        return ap_result_str, ap_dict


def create_robosense_infos(dataset_cfg, data_path, save_path):
    import os
    num_workers = 4

    train_scenes = ['ruby119_nanshandadao_1200421163451', 'ruby_ruby002_baoshenlu_1200303103447', 'ruby_ruby136_shizilukou_1200526171538',
                    'ruby_ruby144_shizilukou_1200529160951']
    val_scenes = ['ruby112_lishanlu_1200430192539', 'ruby119_longzhudadao_1200423181920', 'ruby_ruby002_wushitoulu_1200303111734',
                  'ruby_ruby136_shizilukou_1200521120824', 'ruby_ruby136_shizilukou_1200526161859']

    # 所有pcd 和 label路径
    train_data_path = []
    train_label_path = []
    val_data_path = []
    val_label_path = []
    for scene in train_scenes:
        scene_label_path = os.listdir(data_path / scene / 'label')
        for path in scene_label_path:
            name = path.split('.')[0]
            train_data_path.append(data_path / scene / 'npy' / (name + '.npy'))
            train_label_path.append(data_path / scene / 'label' / (name + '.json'))

    for scene in val_scenes:
        scene_label_path = os.listdir(data_path / scene / 'label')
        for path in scene_label_path:
            name = path.split('.')[0]
            val_data_path.append(data_path / scene / 'npy' / (name + '.npy'))
            val_label_path.append(data_path / scene / 'label' / (name + '.json'))

    from pcdet.datasets.robosense import robosense_utils
    import concurrent.futures as futures
    with futures.ThreadPoolExecutor(num_workers) as executor:
        train_infos = executor.map(robosense_utils.process_single_data, zip(train_data_path, train_label_path))
    with futures.ThreadPoolExecutor(num_workers) as executor:
        val_infos = executor.map(robosense_utils.process_single_data, zip(val_data_path, val_label_path))

    train_file_name = save_path / 'robosense_infos_train.pkl'
    val_file_name = save_path / 'robosense_infos_val.pkl'
    with open(train_file_name, 'wb') as f:
        pickle.dump(list(train_infos), f)
    with open(val_file_name, 'wb') as f:
        pickle.dump(list(val_infos), f)

    robosense_utils.create_groundtruth_database(info_path=train_file_name, save_path=save_path)
    print('---------------Data preparation Done---------------')


# test for debug by ck
if __name__ == '__main__':
    import sys
    import os
    import yaml
    import json
    import datetime
    from pathlib import Path
    from easydict import EasyDict
    print(os.path.abspath('../../..')+'/tools/cfgs/dataset_configs/robosense_dataset.yaml')
    dataset_cfg = EasyDict(yaml.load(open(os.path.abspath('../../..')+'/tools/cfgs/dataset_configs/robosense_dataset.yaml')))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    ROOT_DIR_ck = Path(dataset_cfg.DATA_PATH)
    create_robosense_infos(
        dataset_cfg=dataset_cfg,
        data_path=ROOT_DIR_ck,
        save_path=ROOT_DIR_ck
    )

    # log_file = ROOT_DIR_ck / 'data_log' / ('log_data_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    # logger = common_utils.create_logger(log_file)
    # robosense_dataset = RobosenseDataset(
    #     dataset_cfg=dataset_cfg, class_names=['vehicle', 'pedestrian', 'bicycle'],
    #     root_path=ROOT_DIR_ck,
    #     logger=logger, training=True
    # )
    #
    # one_sample = robosense_dataset.__getitem__(8)
    # print('RoboSense Read data end!')
