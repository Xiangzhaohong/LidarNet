import numpy as np
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pathlib import Path
import json
import torch

# map_name_from_general_to_detection = {
#     'pedestrian': 'pedestrian',
#     'bicycle': 'bicycle',
#     'motorcycle': 'bicycle',
#     'unknown': 'unknown',
#     'cone': 'unknown',
#     'tricycle': 'unknown',
#     'vehicle': 'vehicle',
#     'big_vehicle': 'unknown',
#     'huge_vehicle': 'unknown',
# }

# for multi head
map_name_from_general_to_detection = {
    'pedestrian': 'pedestrian',
    'bicycle': 'bicycle',
    'motorcycle': 'motorcycle',
    'unknown': 'unknown',
    'cone': 'cone',
    'tricycle': 'tricycle',
    'vehicle': 'vehicle',
    'big_vehicle': 'big_vehicle',
    'huge_vehicle': 'huge_vehicle',
}


def load_pcd_to_ndarray(pcd_path):
    # import pcl
    name = Path(pcd_path).stem + '.npy'
    cur_save_path = Path(pcd_path).parent.parent / 'npy' / name
    # cur_save_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(str(cur_save_path), pc.astype(np.float32))
    pc = np.load(str(cur_save_path))
    # pc = pcl.load_XYZI(cur_save_path.__str__()).to_array()[:, :4]
    pc = np.delete(pc, np.where(np.isnan(pc[:, 0]))[0], axis=0)

    # lidar = []
    # with open(pcd_path.__str__(), 'rb') as f:
    #     for i in range(11):
    #         next(f)
    #     line = f.readline().strip()
    #     while line:
    #         linestr = line.split(" ")
    #         if len(linestr) == 4:
    #             linestr_convert = list(map(float, linestr))
    #             lidar.append(linestr_convert)
    #         line = f.readline().strip()
    # return np.array(lidar)
    # name = Path(pcd_path).stem #+ '.npy'
    # cur_save_path = Path(pcd_path).parent.parent / 'npy' / name
    # cur_save_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(str(cur_save_path), pc.astype(np.float32))
    return pc.astype(np.float32)


def get_label(label_path):
    assert label_path.exists()
    obj_list = []
    labels = json.load(open(label_path))
    for label in labels['labels']:
        obj = {}
        obj['loc'] = np.array((float(label['center']['x']), float(label['center']['y']), float(label['center']['z'])), dtype=np.float32)
        obj['size'] = np.array((float(label['size']['x']), float(label['size']['y']), float(label['size']['z'])), dtype=np.float32)
        # obj['in_roi'] = bool(label['in_roi'])
        # obj['is_valid'] = bool(label['is_valid'])
        obj['yaw'] = float(label['rotation']['yaw'])
        obj['tracker_id'] = int(label['tracker_id'])
        obj['type'] = label['type']
        # obj['visibility'] = label['visibility']

        obj_list.append(obj)

    return obj_list


def process_single_data(sample_path=None):
    sample_data_path, sample_label_path = sample_path
    info = {}
    info['lidar_path'] = str(sample_data_path)
    info['label_path'] = str(sample_label_path)
    frame_id = sample_data_path.stem.split('_')[-1]
    scene_name = str(sample_data_path.parent.parent.stem)
    # info['frame_id'] = scene_name + ':' + frame_id
    # info['image'] = None
    # info['calib'] = None
    if sample_label_path is not None:
        obj_list = get_label(sample_label_path)
        lidar_points = load_pcd_to_ndarray(info['lidar_path'])
        annotations = {}
        # TODO
        # 不应该在这变换name的,这样每次就要重新生成数据
        # not use abs path better!
        annotations['gt_names'] = np.array([obj['type'] for obj in obj_list])
        annotations['location'] = np.concatenate([obj['loc'].reshape(1, 3) for obj in obj_list], axis=0)
        annotations['size'] = np.concatenate([obj['size'].reshape(1, 3) for obj in obj_list], axis=0)
        annotations['yaw'] = np.array([obj['yaw'] for obj in obj_list], dtype=np.float32)
        # 要不要在这去掉don't care类?
        # ==>不要吧,只加载原始数据,其他不做处理
        gt_boxes_lidar = np.concatenate(
            [annotations['location'], annotations['size'], annotations['yaw'][..., np.newaxis]], axis=1)
        annotations['gt_boxes_lidar'] = gt_boxes_lidar

        # 计算每个box里面的点数,为后面训练和测试时过滤掉点数很少的box做准备
        gt_boxes_PointsNum = []
        point_indices = roiaware_pool3d_utils.points_in_boxes_gpu(
            torch.from_numpy(lidar_points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
            torch.from_numpy(gt_boxes_lidar).unsqueeze(dim=0).float().cuda()
        ).long().squeeze(dim=0).cpu().numpy()  # (nboxes, npoints)
        for i in range(gt_boxes_lidar.shape[0]):
            gt_points = lidar_points[point_indices == i]
            gt_boxes_PointsNum.append(gt_points.shape[0])
        annotations['boxes_points_pts'] = np.array(gt_boxes_PointsNum, dtype=np.int)

        annotations['num_lidar_pts'] = lidar_points.shape[0]
        annotations['frame_id'] = scene_name + ':' + frame_id
        info['annos'] = annotations

    return info


def create_groundtruth_database(info_path=None, save_path=None, used_classes=None):
    import pickle
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    database_save_path = Path(save_path) / 'gt_database'
    db_info_save_path = Path(save_path) / 'robosense_dbinfos_train.pkl'
    database_save_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    for k in range(len(infos)):
        print('gt_database sample: %d/%d' % (k + 1, len(infos)))
        info = infos[k]
        lidar_points = load_pcd_to_ndarray(info['lidar_path'])
        annos = info['annos']
        names = annos['gt_names']
        gt_boxes = annos['gt_boxes_lidar']

        num_obj = gt_boxes.shape[0]
        point_indices = roiaware_pool3d_utils.points_in_boxes_gpu(
            torch.from_numpy(lidar_points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
            torch.from_numpy(gt_boxes).unsqueeze(dim=0).float().cuda()
        ).long().squeeze(dim=0).cpu().numpy()  # (nboxes, npoints)

        # point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
        #     torch.from_numpy(lidar_points[:, 0:3]), torch.from_numpy(gt_boxes)
        # ).numpy()  # (nboxes, npoints)

        for i in range(num_obj):
            filename = '%d_%s_%d.bin' % (k, names[i], i)
            filepath = database_save_path / filename
            # gt_points = lidar_points[point_indices[i] > 0]
            gt_points = lidar_points[point_indices == i]

            # 相对坐标
            gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_path = str(filepath.relative_to(save_path))
                db_info = {
                    'name': names[i], 'path': db_path, 'image_idx': k, 'gt_idx': i,
                    'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]
                }
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print('Database %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['gt_names']) if x != name]
    for key in info.keys():
        if isinstance(info[key], np.ndarray):
            ret_info[key] = info[key][keep_indices]
    return ret_info


def drop_info_with_box_points(info, min_pts):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['boxes_points_pts']) if x >= min_pts]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


# --------------plot for test-----------------------
def visual_pcd_box(lidar_path=None, label_path=None):
    if label_path is not None:
        label = json.load(open(label_path))
    else:
        # for test
        label = json.load(open(
            '/home/syang/Data/RS_datasets/datasets/ruby112_lishanlu_1200430192539/label/ruby112_lishanlu_1200430192539_1405.json'))
    label = label['labels']

    if lidar_path is not None:
        cloud_points = load_pcd_to_ndarray(lidar_path)
    else:
        # for test
        cloud_points = load_pcd_to_ndarray(
            '/home/syang/Data/RS_datasets/datasets/ruby112_lishanlu_1200430192539/pcd/ruby112_lishanlu_1200430192539_1405.pcd')

    from tools.visual_utils.visualize_utils import draw_scenes
    import mayavi.mlab as mlab

    # boxes3d: (N, 7)[x, y, z, dx, dy, dz, heading]
    boxes = []
    gt_labels = []

    for i in range(len(label)):
        cur_obj = label[i]
        boxes.append([cur_obj['center']['x'], cur_obj['center']['y'], cur_obj['center']['z'],
                      cur_obj['size']['x'], cur_obj['size']['y'], cur_obj['size']['z'],
                      cur_obj['rotation']['yaw']])
        gt_labels.append(cur_obj['type'])
    gt_boxes = np.array(boxes)
    draw_scenes(points=cloud_points, gt_boxes=gt_boxes, ref_boxes=None, ref_scores=None, ref_labels=None,
                gt_labels=gt_boxes[:, 6])
    mlab.show(stop=True)
    #mlab.savefig('test.png')


def plot_examples(lidar_path=None, label_path=None):
    if label_path is not None:
        label = json.load(open(label_path))
    else:
        # for test
        label = json.load(open(
            '/home/syang/Data/RS_datasets/datasets/ruby112_lishanlu_1200430192539/label/ruby112_lishanlu_1200430192539_1405.json'))
    label = label['labels']

    if lidar_path is not None:
        cloud_points = load_pcd_to_ndarray(lidar_path)
    else:
        # for test
        cloud_points = load_pcd_to_ndarray(
            '/home/syang/Data/RS_datasets/datasets/ruby112_lishanlu_1200430192539/pcd/ruby112_lishanlu_1200430192539_1405.pcd')

    from tools.visual_utils.visualize_utils import draw_scenes
    from matplotlib import pyplot as plt

    # boxes3d: (N, 7)[x, y, z, dx, dy, dz, heading]
    boxes = []
    gt_labels = []

    for i in range(len(label)):
        cur_obj = label[i]
        boxes.append([cur_obj['center']['x'], cur_obj['center']['y'], cur_obj['center']['z'],
                      cur_obj['size']['x'], cur_obj['size']['y'], cur_obj['size']['z'],
                      cur_obj['rotation']['yaw']])
        gt_labels.append(cur_obj['type'])
    gt_boxes = np.array(boxes)
    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    # Show point cloud.
    ax.scatter(cloud_points[:, 0], cloud_points[:, 1], s=0.2)
    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')
    # Show GT boxes.
    plt.savefig('test_plt.png')
    plt.close()


if __name__ == '__main__':
    # visual_pcd_box()
    # print('test')
    load_pcd_to_ndarray('/home/syang/Data/RS_datasets/datasets/ruby112_lishanlu_1200430192539/pcd/ruby112_lishanlu_1200430192539_1405.pcd')
