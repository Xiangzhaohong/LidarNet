from pathlib import Path
import numpy as np
import numba
# from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.datasets.robosense import robosense_utils
from matplotlib import pyplot as plt
from nuscenes.eval.common.render import setup_axis
from pcdet.utils import box_utils, common_utils

# TODO
# jit函数 在类中定义就报错
@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def bev_box_overlap(gt_boxes, dt_boxes):
    # gt_boxes/dt_boxes   (num_box, 7)
    gt_boxes_bev = gt_boxes
    dt_boxes_bev = dt_boxes

    riou = iou3d_nms_utils.boxes_bev_iou_cpu(gt_boxes_bev, dt_boxes_bev)

    return riou


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_bboxes = dt_datas[:, :-1]
    gt_bboxes = gt_datas[:, :-1]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn = 0, 0, 0
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[i, j]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        # if metric == 0:
        #     overlaps_dt_dc = bev_box_overlap(dt_bboxes, dc_bboxes)
        #     for k in range(dc_bboxes.shape[0]):
        #         for j in range(det_size):
        #             if (assigned_detection[j]):
        #                 continue
        #             if (ignored_det[j] == -1 or ignored_det[j] == 1):
        #                 continue
        #             if (ignored_threshold[j]):
        #                 continue
        #             if overlaps_dt_dc[j, k] > min_overlap:
        #                 assigned_detection[j] = True
        #                 nstuff += 1
        # fp -= nstuff
    return tp, fp, fn, thresholds[:thresh_idx]


@numba.jit(nopython=True)
def fused_compute_statistics(
                             overlaps,
                             pr,
                             gt_nums, dt_nums, dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[gt_num:gt_num + gt_nums[i],
                                dt_num:dt_num + dt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn

        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


class RoboSense_Eval:
    def __init__(self, gt_annos, dt_annos, current_classes, output_dir,
                 min_overlaps=None, class_to_name=None):
        self.gt_annos = gt_annos
        self.dt_annos = dt_annos

        self.output_dir = Path(output_dir)
        self.plot_dir = self.output_dir / 'plot'
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.plot_dir.exists():
            self.plot_dir.mkdir(parents=True, exist_ok=True)

        name_to_class = {v: n for n, v in class_to_name.items()}
        if not isinstance(current_classes, (list, tuple)):
            current_classes = [current_classes]
        current_classes_int = []
        for curcls in current_classes:
            if isinstance(curcls, str):
                current_classes_int.append(name_to_class[curcls])
            else:
                current_classes_int.append(curcls)
        self.current_classes = current_classes
        self.current_classes_int = current_classes_int
        self.class_to_name = class_to_name
        self.name_to_class = name_to_class

        if min_overlaps is not None:
            # 保留原kitti的形式(num_minoverlap, metric(对应不同的box iou种类), num_class)
            if len(min_overlaps.shape) == 1:
                min_overlaps = min_overlaps[np.newaxis, np.newaxis, :]
            if len(min_overlaps.shape) == 2:
                min_overlaps = min_overlaps[np.newaxis, :]
            min_overlaps = min_overlaps[:, :, current_classes_int]
            self.min_overlaps = min_overlaps   # (2, 3, 5)
        else:
            self.min_overlaps = None

    def image_box_overlap(self, gt_boxes, dt_boxes):
        pass

    def d3_box_overlap(self, gt_boxes, dt_boxes):
        pass

    def get_split_parts(self, num, num_part):
        same_part = num // num_part
        remain_num = num % num_part
        if same_part == 0:
            return [num]

        if remain_num == 0:
            return [same_part] * num_part
        else:
            return [same_part] * num_part + [remain_num]

    def calculate_iou_partly(self, metric, num_parts=50):
        """fast iou algorithm. this function can be used independently to
        do result analysis. Must be used in CAMERA coordinate system.
        Args:
            gt_annos: dict, must from get_label_annos() in kitti_common.py
            dt_annos: dict, must from get_label_annos() in kitti_common.py
            metric: eval type. 0: bbox, 1: bev, 2: 3d
            num_parts: int. a parameter for fast calculate algorithm
        """
        total_dt_num = np.stack([len(a['name']) for a in self.dt_annos], 0)
        total_gt_num = np.stack([len(a['gt_names']) for a in self.gt_annos], 0)
        num_examples = len(self.gt_annos)
        split_parts = self.get_split_parts(num_examples, num_parts)
        parted_overlaps = []
        example_idx = 0

        for num_part in split_parts:
            gt_annos_part = self.gt_annos[example_idx:example_idx + num_part]
            dt_annos_part = self.dt_annos[example_idx:example_idx + num_part]
            if metric == 2:
                gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
                dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
                overlap_part = self.image_box_overlap(gt_boxes, dt_boxes)
            elif metric == 0:
                gt_boxes = np.concatenate(
                    [a['gt_boxes_lidar'] for a in gt_annos_part], 0)

                dt_boxes = np.concatenate(
                    [a['boxes_lidar'] for a in dt_annos_part], 0)

                overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
            elif metric == 1:
                loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
                dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
                gt_boxes = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1)
                loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
                dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1)
                overlap_part = self.d3_box_overlap(gt_boxes, dt_boxes).astype(
                    np.float64)
            else:
                raise ValueError("unknown metric")
            parted_overlaps.append(overlap_part)
            example_idx += num_part
        overlaps = []
        example_idx = 0

        for j, num_part in enumerate(split_parts):
            gt_num_idx, dt_num_idx = 0, 0
            for i in range(num_part):
                gt_box_num = total_gt_num[example_idx + i]
                dt_box_num = total_dt_num[example_idx + i]
                overlaps.append(
                    parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                    dt_num_idx:dt_num_idx + dt_box_num])
                gt_num_idx += gt_box_num
                dt_num_idx += dt_box_num
            example_idx += num_part

        return overlaps, parted_overlaps, total_gt_num, total_dt_num

    def clean_data(self, gt_anno, dt_anno, current_class, difficulty):
        num_gt = len(gt_anno['gt_names'])
        num_dt = len(dt_anno['name'])
        dc_bboxes, ignored_gt, ignored_dt = [], [], []
        num_valid_gt = 0

        for i in range(num_gt):
            if gt_anno['gt_names'][i] == current_class:
                valid_class = 1
            else:
                valid_class = -1

            if valid_class == 1:
                ignored_gt.append(0)
                num_valid_gt += 1
            else:
                ignored_gt.append(-1)

            if gt_anno['gt_names'][i] == 'unknown':
                dc_bboxes.append(gt_anno['gt_boxes_lidar'][i, 0:7])

        for i in range(num_dt):
            if dt_anno['name'][i] == current_class:
                valid_class = 1
            else:
                valid_class = -1
            if valid_class == 1:
                ignored_dt.append(0)
            else:
                ignored_dt.append(-1)

        return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes

    def _prepare_data(self, current_class, difficulty):
        gt_datas_list = []
        dt_datas_list = []
        total_dc_num = []
        ignored_gts, ignored_dets, dontcares = [], [], []
        total_num_valid_gt = 0

        for i in range(len(self.gt_annos)):
            rets = self.clean_data(self.gt_annos[i], self.dt_annos[i], current_class, difficulty)
            num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets

            ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
            ignored_dets.append(np.array(ignored_det, dtype=np.int64))
            if len(dc_bboxes) == 0:
                dc_bboxes = np.zeros((0, 7)).astype(np.float64)
            else:
                dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
            total_dc_num.append(dc_bboxes.shape[0])
            dontcares.append(dc_bboxes)
            total_num_valid_gt += num_valid_gt

            gt_datas = self.gt_annos[i]["gt_boxes_lidar"]
            dt_datas = np.concatenate([
                self.dt_annos[i]["boxes_lidar"], self.dt_annos[i]["score"][..., np.newaxis]
            ], 1)
            gt_datas_list.append(gt_datas)
            dt_datas_list.append(dt_datas)

        total_dc_num = np.stack(total_dc_num, axis=0)
        return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
                total_dc_num, total_num_valid_gt)

    def get_mAP(self, prec):
        sums = 0
        for i in range(0, prec.shape[-1], 4):
            sums = sums + prec[..., i]
        return sums / 11 * 100

    def get_mAP_R40(self, prec):
        sums = 0
        for i in range(1, prec.shape[-1]):
            sums = sums + prec[..., i]
        return sums / 40 * 100

    def examples_plot(self, plot_multi=0, plot_examples=None):
        import mayavi.mlab as mlab
        from tools.visual_utils.visualize_utils import draw_scenes
        plot_examples = 1
        data_ROOT_Path = '/home/syang/Data/RS_datasets/datasets'
        if plot_multi:
            np.random.seed(42)
            sample_tokens = np.random.randint(0, len(self.dt_annos), plot_examples)
            for sample in list(sample_tokens):
                det_anno = self.dt_annos[sample]
                gt_anno = self.gt_annos[sample]
                frame_id = str(gt_anno['frame_id'])
                pcd_dir = Path(data_ROOT_Path) / frame_id.split(':')[0] / 'pcd' /\
                          (frame_id.split(':')[0] + '_' + frame_id.split(':')[1] + '.pcd')

                lidar_points = robosense_utils.load_pcd_to_ndarray(str(pcd_dir))
                draw_scenes(points=lidar_points, gt_boxes=gt_anno['gt_boxes_lidar'], ref_boxes=det_anno['boxes_lidar'],
                            ref_scores=det_anno['score'], ref_labels=det_anno['pred_labels'])
                save_scene_path = self.plot_dir / (str(sample) + '.png')
                # mlab.show()
                mlab.savefig(str(save_scene_path))
        else:
            det_anno = self.dt_annos[plot_examples]
            gt_anno = self.gt_annos[plot_examples]
            frame_id = str(gt_anno['frame_id'])
            pcd_dir = Path(data_ROOT_Path) / frame_id.split(':')[0] / 'pcd' / \
                      (frame_id.split(':')[0] + '_' + frame_id.split(':')[1] + '.pcd')

            lidar_points = robosense_utils.load_pcd_to_ndarray(str(pcd_dir))
            draw_scenes(points=lidar_points, gt_boxes=gt_anno['gt_boxes_lidar'], ref_boxes=det_anno['boxes_lidar'],
                        ref_scores=det_anno['score'], ref_labels=det_anno['pred_labels'])
            save_scene_path = self.plot_dir / (str(plot_examples) + '.png')
            mlab.show()

    def evaluate(self):
        # use IOU, for lidar, here only use bev rotate iou
        if self.min_overlaps is not None:
            num_examples = len(self.gt_annos)
            num_parts = 100
            N_SAMPLE_PTS = 41
            difficultys = [0]
            metric = 0  # only use bev rotate iou
            ret_dict = {}

            split_parts = self.get_split_parts(num_examples, num_parts)
            rets = self.calculate_iou_partly(metric=metric, num_parts=num_parts)
            overlaps, parted_overlaps, total_gt_num, total_dt_num = rets

            num_minoverlap = len(self.min_overlaps)
            num_class = len(self.current_classes)
            num_difficulty = len(difficultys)

            precision = np.zeros(
                [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
            recall = np.zeros(
                [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
            score_thresholds = np.zeros(
                [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
            )

            for m, current_class in enumerate(self.current_classes):
                print('eval class %s' % current_class)
                for l, difficulty in enumerate(difficultys):
                    rets = self._prepare_data(current_class, difficulty)
                    (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
                     dontcares, total_dc_num, total_num_valid_gt) = rets

                    for k, min_overlap in enumerate(self.min_overlaps[:, metric, m]):
                        thresholdss = []
                        for i in range(len(self.gt_annos)):
                            rets = compute_statistics_jit(
                                overlaps[i],
                                gt_datas_list[i],
                                dt_datas_list[i],
                                ignored_gts[i],
                                ignored_dets[i],
                                dontcares[i],
                                metric,
                                min_overlap=min_overlap,
                                thresh=0.0,
                                compute_fp=False)
                            tp, fp, fn, thresholds = rets
                            thresholdss += thresholds.tolist()

                        thresholdss = np.array(thresholdss)
                        thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                        thresholds = np.array(thresholds)
                        pr = np.zeros([len(thresholds), 4])
                        idx = 0

                        for j, num_part in enumerate(split_parts):
                            gt_datas_part = np.concatenate(
                                gt_datas_list[idx:idx + num_part], 0)
                            dt_datas_part = np.concatenate(
                                dt_datas_list[idx:idx + num_part], 0)
                            dc_datas_part = np.concatenate(
                                dontcares[idx:idx + num_part], 0)
                            ignored_dets_part = np.concatenate(
                                ignored_dets[idx:idx + num_part], 0)
                            ignored_gts_part = np.concatenate(
                                ignored_gts[idx:idx + num_part], 0)

                            fused_compute_statistics(
                                parted_overlaps[j],
                                pr,
                                total_gt_num[idx:idx + num_part],
                                total_dt_num[idx:idx + num_part],
                                total_dc_num[idx:idx + num_part],
                                gt_datas_part,
                                dt_datas_part,
                                dc_datas_part,
                                ignored_gts_part,
                                ignored_dets_part,
                                metric,
                                min_overlap=min_overlap,
                                thresholds=thresholds)
                            idx += num_part

                        # score_thresholds[m, l, k, :] = thresholds
                        for i in range(len(thresholds)):
                            recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                            precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])

                        # for i in range(len(thresholds)):
                        #     precision[m, l, k, i] = np.max(
                        #         precision[m, l, k, i:], axis=-1)
                        #     recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)

            mAP_bev = self.get_mAP(precision)
            mAP_bev_R40 = self.get_mAP_R40(precision)

            # plot P-R
            for m, current_class in enumerate(self.current_classes):
                ax = setup_axis(title='P-R', xlabel='Recall', ylabel='Precision', xlim=1, ylim=1)
                ax.plot(recall[m, 0, 0, :], precision[m, 0, 0, :], '.',
                        label='P-R. : {}, AP: {:.1f}'.format(current_class, mAP_bev_R40[m, 0, 0]))

                ax.legend(loc='best')
                ret_dict['%s_bev/R40' % current_class] = mAP_bev_R40[m, 0, 0]

            savepath = str(self.plot_dir / ('PR' + '.png'))
            plt.savefig(savepath)
            plt.close()

            print('mAP_bev %.4f, %.4f, %.4f' % (mAP_bev[0, 0], mAP_bev[1, 0], mAP_bev[2, 0]))
            print('mAP_bev_R40 %.4f, %.4f, %.4f' % (mAP_bev_R40[0, 0], mAP_bev_R40[1, 0], mAP_bev_R40[2, 0]))
            result_str = ''
            return result_str, ret_dict


def test_eval(follow_kitti=False):
    import pickle
    from pathlib import Path
    gt_datas_dir = '/home/syang/Projects/OpenPCDet/output/robosense_models/robosense_pointpillar/test_1214_RS/eval/epoch_80/val/default/final_result/data/input_batch_dict.pkl'
    dt_datas_dir = '/home/syang/Projects/OpenPCDet/output/robosense_models/robosense_pointpillar/test_1214_RS/eval/epoch_80/val/default/final_result/data/output_batch_dict.pkl'

    class_names = ['vehicle', 'pedestrian', 'bicycle']
    ROOT_DIR_ck = Path('/home/syang/Data/RS_datasets/datasets')

    with open(gt_datas_dir, 'rb') as f:
        gt_infos = pickle.load(f)
    with open(dt_datas_dir, 'rb') as f:
        dt_infos = pickle.load(f)
    eval_det_annos = []
    eval_gt_annos = []
    for i in range(len(dt_infos)):
        batch_dt_info = dt_infos[i]
        for j in range(len(batch_dt_info)):
            dt_info = batch_dt_info[j]

            nora_dt_info = {}
            pred_scores = dt_info['pred_scores'].cpu().numpy()
            pred_boxes = dt_info['pred_boxes'].cpu().numpy()
            pred_labels = dt_info['pred_labels'].cpu().numpy()

            nora_dt_info['name'] = np.array(class_names)[pred_labels.astype(np.int) - 1]
            nora_dt_info['score'] = pred_scores
            nora_dt_info['boxes_lidar'] = pred_boxes
            nora_dt_info['pred_labels'] = pred_labels
            eval_det_annos.append(nora_dt_info)

    for i in range(len(gt_infos)):
        batch_gt_info = gt_infos[i]
        for j in range(batch_gt_info['gt_boxes'].shape[0]):
            gt_boxes = batch_gt_info['gt_boxes'][j, :, :]   #(n, 8)
            mask = gt_boxes.sum(axis=1) != 0
            gt_boxes = gt_boxes[mask]

            nora_gt_info = {}
            nora_gt_info['gt_names'] = np.array(class_names)[gt_boxes[:, -1].astype(np.int) - 1]
            nora_gt_info['name'] = np.array(class_names)[gt_boxes[:, -1].astype(np.int) - 1]
            nora_gt_info['gt_boxes_lidar'] = gt_boxes[:, 0:-1]
            nora_gt_info['frame_id'] = batch_gt_info['frame_id'][j]
            eval_gt_annos.append(nora_gt_info)

    #Filter boxes (distance, points per box, etc.).
    point_cloud_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    filtered_gt_box_num_By_range = 0
    for i in range(len(eval_gt_annos)):
        mask = box_utils.mask_boxes_outside_range_numpy(eval_gt_annos[i]['gt_boxes_lidar'],
                                                        point_cloud_range)
        filtered_gt_box_num_By_range += (eval_gt_annos[i]['gt_boxes_lidar'].shape[0] - mask.sum())
        for key in eval_gt_annos[i].keys():
            if isinstance(eval_gt_annos[i][key], np.ndarray):
                eval_gt_annos[i][key] = eval_gt_annos[i][key][mask]
    print('Eval preprocess--filter gt box by PointCloud range. filtered %d GT boxes' % (
        int(filtered_gt_box_num_By_range)))

    #filtered_dt_box_num_By_score
    min_score = 0.2
    filtered_det_box_num_By_score = 0
    for i in range(len(eval_det_annos)):
        keep_indices = [i for i, x in enumerate(eval_det_annos[i]['score']) if x >= min_score]
        filtered_det_box_num_By_score += (len(eval_det_annos[i]['score']) - len(keep_indices))
        for key in eval_det_annos[i].keys():
            if isinstance(eval_det_annos[i][key], np.ndarray):
                eval_det_annos[i][key] = eval_det_annos[i][key][keep_indices]
    print('Eval preprocess--filter det box by detection score. filtered %d GT boxes' % (
        int(filtered_det_box_num_By_score)))

    if follow_kitti:
        from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
        from pcdet.datasets.kitti import kitti_utils

        map_name_to_kitti = {
            'vehicle': 'Car',
            'pedestrian': 'Pedestrian',
            'bicycle': 'Cyclist',
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
        # plot P-R
        # [recall]:[num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
        for m, current_class in enumerate(kitti_class_names):
            mAP_R40 = ap_dict['%s_bev/easy_R40' % current_class]
            ax = setup_axis(title='P-R_KITTI', xlabel='Recall', ylabel='Precision', xlim=1, ylim=1)
            ax.plot(PR_detail_dict['bev-recall'][m, 0, 0, :],
                    PR_detail_dict['bev'][m, 0, 0, :], '.',
                    label='P-R. : {}, AP: {:.1f}'.format(current_class, mAP_R40))

            ax.legend(loc='best')

        savepath = str(ROOT_DIR_ck / 'plot' / ('KITTI_PR_12' + '.png'))
        plt.savefig(savepath)
        plt.close()

    else:
        overlap_0_7 = np.array([0.7, 0.5, 0.5, 0.7, 0.5, 0.7])
        min_overlaps = overlap_0_7
        class_to_name = {
            0: 'vehicle',
            1: 'pedestrian',
            2: 'bicycle'
        }

        my_eval = RoboSense_Eval(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos,
            current_classes=class_names, output_dir=ROOT_DIR_ck,
            class_to_name=class_to_name, min_overlaps=min_overlaps
        )
        my_eval.examples_plot(plot_multi=0, plot_examples=45)

        my_eval.evaluate()

    print('EVAL!')


# TODO
if __name__ == '__main__':
    from pcdet.datasets.robosense.robosense_dataset import RobosenseDataset
    import yaml
    import json
    import datetime
    from pathlib import Path
    from easydict import EasyDict
    import torch

    test_eval(follow_kitti=True)

    dataset_cfg = EasyDict(
        yaml.load(open('/home/syang/Projects/OpenPCDet/tools/cfgs/dataset_configs/robosense_dataset.yaml')))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    ROOT_DIR_ck = Path('/home/syang/Data/RS_datasets/datasets')

    log_file = ROOT_DIR_ck / 'data_log' / ('log_data_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file)
    # test val
    from pcdet.datasets import build_dataloader

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=dataset_cfg,
        class_names=['vehicle', 'pedestrian', 'bicycle'],
        root_path=ROOT_DIR_ck,
        batch_size=4,
        dist=False, workers=1, logger=logger, training=False
    )

    dataset = test_loader.dataset
    class_names = dataset.class_names
    det_annos = []
    aa = iter(test_loader)
    for i in range(10):
        batch_dict = aa.__next__()
        print(i)
        if i == 10:
            break
        annos = batch_dict
        pred_dicts = []
        for k in range(batch_dict['gt_boxes'].shape[0]):
            mask = batch_dict['gt_boxes'][k].sum(axis=1) != 0
            pred_fake = {}
            score = np.random.randint(low=6, high=10, size=mask.shape[0])/10
            score = score[mask]
            score = torch.from_numpy(score).cuda()
            pred_fake['pred_scores'] = score
            pred_fake['pred_boxes'] = torch.from_numpy(batch_dict['gt_boxes'][k][mask][:, 0:7]).cuda()
            pred_fake['pred_labels'] = torch.from_numpy(batch_dict['gt_boxes'][k][mask][:, 7:8]).cuda()

            pred_dicts.append(pred_fake)

        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=ROOT_DIR_ck
        )
        det_annos += annos

    output_path = ROOT_DIR_ck
    output_path_for_eval = Path(output_path)
    res_path = str(output_path / 'results_robosense.pkl')
    import pickle
    with open(res_path, 'wb') as f:
        pickle.dump(list(det_annos), f)
    print()
    dataset.evaluation(
        det_annos, class_names
    )
    pass