# ros package
import rospy
import ros_numpy
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point

from std_msgs.msg import Header

import numpy as np
import os
import sys
import torch
import time
import glob

from pathlib import Path
from pcdet.datasets import DatasetTemplate
from pyquaternion import Quaternion
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path/ f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfil(self.sample_file_list[index], dtype=np.float32).reshape(-1,4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)


def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices


def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["pred_labels"].detach().cpu().numpy()
    scores_ = image_anno["pred_scores"].detach().cpu().numpy()

    cat_indices = get_annotations_indices(1, 0.55, label_preds_, scores_)
    truck_indices = get_annotations_indices(2, 0.50, label_preds_, scores_)
    costruction_vehicle_indices = get_annotations_indices(3, 0.15, label_preds_, scores_)
    bus_indices = get_annotations_indices(4, 0.15, label_preds_, scores_)
    trailer_indices = get_annotations_indices(5, 0.15, label_preds_, scores_)
    barrier_indices = get_annotations_indices(6, 0.15, label_preds_, scores_)
    motorcycle_indices = get_annotations_indices(7, 0.15, label_preds_, scores_)
    bicycle_indices = get_annotations_indices(8, 0.15, label_preds_, scores_)
    pedestrain_indices = get_annotations_indices(9, 0.15, label_preds_, scores_)
    traffic_cone_indices = get_annotations_indices(10, 0.15, label_preds_, scores_)

    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key]=(
            image_anno[key][cat_indices+
                            pedestrain_indices+
                            bicycle_indices+
                            bus_indices+
                            traffic_cone_indices+
                            costruction_vehicle_indices+
                            trailer_indices+
                            barrier_indices+
                            motorcycle_indices+
                            truck_indices])

    return img_filtered_annotations


class Precessor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None

    def initialize(self):
        self.read_config()

    def read_config(self):
        config_path = self.config_path
        cfg_from_yaml_file(self.config_path,cfg)
        self.logger = common_utils.create_logger()
        self.demo_datasets  = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path('/home/xjtuiair/KITTI/KITTI/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000097.bin'),
            ext='.bin'
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_datasets)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

    def run(self, points):
        t1 = time.time()
        print(f"input points shape: {points.shape}")
        num_features = 4 #kitti model
        #num_features = 5
        self.points = points.reshape([-1,num_features])

        input_dict = {
            'points': self.points,
            'frame_id': 0,
        }

        data_dict = self.demo_datasets.prepare_data(data_dict=input_dict)
        data_dict = self.demo_datasets.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        torch.cuda.synchronize()
        pred_dicts, _ = self.net.forward(data_dict)
        torch.cuda.synchronize()

        t2 = time.time()
        print(f"net inference cost time: {t2 - t1}")

        pred = remove_low_score_nu(pred_dicts[0], 0.45)
        boxes_lidar = pred['pred_boxes'].detach().cpu().numpy()
        scores = pred['pred_scores'].detach().cpu().numpy()
        types = pred['pred_labels'].detach().cpu().numpy()
        #print(f" pred boxes: { boxes_lidar }")
        #print(f" pred scores: {scores}")
        print(f" pred labels: {types}")

        return scores, boxes_lidar, types


def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (4,), dtype=dtype) # kitti model
    #points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    #points = np.zeros(cloud_array.shape, dtype=dtype)
    #print(points.shape)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z'] + 2.0
    return points


def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    msg = PointCloud2()

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id

    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        #PointField('i', 12, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.array(points_sum, np.float32).tobytes()
    return msg


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    print(is_numpy)
    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def lidar_callback(msg):
    t1 = time.time()
    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    frame_id = msg.header.frame_id
    np_p = get_xyz_points(msg_cloud, True)
    print(np_p.size)
    print("  ")
    scores, dt_box_lidar, types = proc_1.run(np_p)
    bbox_corners3d = boxes_to_corners_3d(dt_box_lidar)

    print(bbox_corners3d.shape)

    empty_markers = MarkerArray()
    clear_marker = Marker()
    clear_marker.header.stamp = rospy.Time.now()
    clear_marker.header.frame_id = "velodyne"
    clear_marker.ns = "objects"
    clear_marker.id = 0
    clear_marker.action = clear_marker.DELETEALL
    clear_marker.lifetime = rospy.Duration()
    empty_markers.markers.append(clear_marker)
    pub_bbox_array.publish(empty_markers)

    bbox_arry = MarkerArray()
    if scores.size != 0:
        for i in range(scores.size):
            point_list = []
            bbox = Marker()
            bbox.type = Marker.LINE_LIST
            bbox.ns = "objects"
            bbox.id = i
            box = bbox_corners3d[i]
            q = yaw2quaternion(float(dt_box_lidar[i][6]))
            for j in range(24):
                p = Point()
                point_list.append(p)

            point_list[0].x = float(box[0, 0])
            point_list[0].y = float(box[0, 1])
            point_list[0].z = float(box[0, 2])
            point_list[1].x = float(box[1, 0])
            point_list[1].y = float(box[1, 1])
            point_list[1].z = float(box[1, 2])

            point_list[2].x = float(box[1, 0])
            point_list[2].y = float(box[1, 1])
            point_list[2].z = float(box[1, 2])
            point_list[3].x = float(box[2, 0])
            point_list[3].y = float(box[2, 1])
            point_list[3].z = float(box[2, 2])

            point_list[4].x = float(box[2, 0])
            point_list[4].y = float(box[2, 1])
            point_list[4].z = float(box[2, 2])
            point_list[5].x = float(box[3, 0])
            point_list[5].y = float(box[3, 1])
            point_list[5].z = float(box[3, 2])

            point_list[6].x = float(box[3, 0])
            point_list[6].y = float(box[3, 1])
            point_list[6].z = float(box[3, 2])
            point_list[7].x = float(box[0, 0])
            point_list[7].y = float(box[0, 1])
            point_list[7].z = float(box[0, 2])

            point_list[8].x = float(box[4, 0])
            point_list[8].y = float(box[4, 1])
            point_list[8].z = float(box[4, 2])
            point_list[9].x = float(box[5, 0])
            point_list[9].y = float(box[5, 1])
            point_list[9].z = float(box[5, 2])

            point_list[10].x = float(box[5, 0])
            point_list[10].y = float(box[5, 1])
            point_list[10].z = float(box[5, 2])
            point_list[11].x = float(box[6, 0])
            point_list[11].y = float(box[6, 1])
            point_list[11].z = float(box[6, 2])

            point_list[12].x = float(box[6, 0])
            point_list[12].y = float(box[6, 1])
            point_list[12].z = float(box[6, 2])
            point_list[13].x = float(box[7, 0])
            point_list[13].y = float(box[7, 1])
            point_list[13].z = float(box[7, 2])

            point_list[14].x = float(box[7, 0])
            point_list[14].y = float(box[7, 1])
            point_list[14].z = float(box[7, 2])
            point_list[15].x = float(box[4, 0])
            point_list[15].y = float(box[4, 1])
            point_list[15].z = float(box[4, 2])

            point_list[16].x = float(box[0, 0])
            point_list[16].y = float(box[0, 1])
            point_list[16].z = float(box[0, 2])
            point_list[17].x = float(box[4, 0])
            point_list[17].y = float(box[4, 1])
            point_list[17].z = float(box[4, 2])

            point_list[18].x = float(box[1, 0])
            point_list[18].y = float(box[1, 1])
            point_list[18].z = float(box[1, 2])
            point_list[19].x = float(box[5, 0])
            point_list[19].y = float(box[5, 1])
            point_list[19].z = float(box[5, 2])

            point_list[20].x = float(box[2, 0])
            point_list[20].y = float(box[2, 1])
            point_list[20].z = float(box[2, 2])
            point_list[21].x = float(box[6, 0])
            point_list[21].y = float(box[6, 1])
            point_list[21].z = float(box[6, 2])

            point_list[22].x = float(box[3, 0])
            point_list[22].y = float(box[3, 1])
            point_list[22].z = float(box[3, 2])
            point_list[23].x = float(box[7, 0])
            point_list[23].y = float(box[7, 1])
            point_list[23].z = float(box[7, 2])

            for j in range(24):
                bbox.points.append(point_list[j])
                #print(f"{j} point {point_list[j]}")
            bbox.scale.x = 0.1
            bbox.color.a = 1.0
            bbox.color.r = 1.0
            bbox.color.g = 0.0
            bbox.color.b = 0.0
            bbox.header.stamp = rospy.Time.now()
            bbox.header.frame_id = "velodyne"
            bbox_arry.markers.append(bbox)

    print("total callback time: ", time.time() - t1)


    pub_bbox_array.publish(bbox_arry)
    cloud_array = xyz_array_to_pointcloud2(np_p[:, :3], rospy.Time.now(), "velodyne")
    print(frame_id)
    pub_point2_.publish(cloud_array)



if __name__ == "__main__":

    global proc

    ## config and model path
    #config_path = '/home/xjtuiair/code/OpenLidarPerceptron/tools/cfgs/kitti_models/second.yaml'
    #modle_path = '/home/xjtuiair/KITTI_PCDet/checkpoint_epoch_80.pth'

    #config_path = '/home/xjtuiair/code/OpenLidarPerceptron/tools/cfgs/kitti_models/pointpillar.yaml'
    #modle_path = '/home/xjtuiair/KITTI_PCDet/pointpillar_7728.pth'
    #config_path = '/home/xjtuiair/code/OpenLidarPerceptron/tools/cfgs/kitti_models/pointpillar.yaml'
    #modle_path = '/home/xjtuiair/KITTI_PCDet/pointpillars_1.pth'
    config_path = '/home/xjtuiair/code/OpenLidarPerceptron/tools/cfgs/robosense_models/pointpillar.yaml'
    modle_path = '/home/xjtuiair/code/OpenLidarPerceptron/models/pointpillar_rob_2.pth'

    #config_path = '/home/xjtuiair/code/OpenLidarPerceptron/tools/cfgs/kitti_models/pv_rcnn.yaml'
    #modle_path = '/home/xjtuiair/KITTI_PCDet/pv_rcnn_8369.pth'

    #config_path = '/home/xjtuiair/code/OpenLidarPerceptron/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml'
    #modle_path = '/home/xjtuiair/KITTI_PCDet/pp_multihead_nds5823.pth'

    proc_1 = Precessor_ROS(config_path, modle_path)

    proc_1.initialize()

    rospy.init_node('net_lidar_ros_node')
    sub_lidar_topic = [
        "/velodyne_points",
        "/rslidar_points",
        "/kitti/velo/pointcloud"
    ]
    sub_ = rospy.Subscriber(sub_lidar_topic[0], PointCloud2, lidar_callback, queue_size=1, buff_size=2**24)

    pub_bbox_array = rospy.Publisher('lidar_net_results', MarkerArray, queue_size=1)
    pub_point2_ = rospy.Publisher('lidar_points', PointCloud2, queue_size=1)

    print("lidar net ros start!")
    rate = rospy.Rate(10)
    rospy.spin()
    rate.sleep()



