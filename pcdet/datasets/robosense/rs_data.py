import numpy as np
import copy
import pickle
import os
import json
import numpy as np
import pcl
import pandas
import sys

from skimage import io

#sys.path.append('~/dataset/OpenPCDet/pcdet/datasets')
from ..dataset import DataAugmentor
#from label_loader_demo_2 import testkkk
#testkkk()

#sys.path.append('..')

#from test_out import testmmm
#testmmm()
#print("测试另一个文件成功了")


#sys.path.insert(0, '..')
#sys.path.append(os.path.abspath('..'))
#from ..datasets import dataset
#sys.path.append('../')
#from dataset import DatasetTemplate
#from ..dataset import DatasetTemplate


class RobosenseDataset(DataAugmentor):

    def __init__(self,dataset_cfg,class_names,training= True, root_path=None,logger = None):
        #参数：配置文件dataset_cfg, 要分类的类名class_names, 是否为训练training= True,
        # 数据集的路径root_path=None,    日志文件logger = None
        # 这里由于是类继承的关系，所以root_path在父类中已经定义
        '''super().__init__(
            dataset_cfg = dataset_cfg,class_names=class_names, 
            training = training, root_path = root_path,logger = logger
        )'''
        
        self.dataset_cfg = dataset_cfg
        self.class_names = class_names
        self.training = training
        self.root_path =root_path
        self.logger =logger
        self.robosense_infos =[]

        pass

    #根据数据地址的路径，获取路径下 文件夹的名字列表
    def get_folder_list(self,root_path):
        folder_list = []
        root_path =root_path
        #读取该目录下所有文件夹的名字，并组成一个列表
        folder_list = os.listdir(root_path)
        return folder_list

    #根据文件夹的列表，返回包含所有文件名的列表 files_list_pcd 和files_list_label 
    def get_files_name_list(self):
        folder_list = []
        folder_list = self.get_folder_list(self.root_path)

        files_list_pcd = []
        files_list_label = []

        for per_folder in folder_list:

            #一条路的文件夹的路径one_road_path
            one_road_path = str(self.root_path+per_folder+'/')
            #一条路下文件夹下的文件列表 one_road_list =['label','pcd']
            one_road_list = self.get_folder_list(one_road_path)

            for one_folder in one_road_list:
                if one_folder == 'pcd':
                    pcd_path = str(one_road_path+one_folder)
                if one_folder == 'label':
                    label_path = str(one_road_path+one_folder)

            #获取pcd文件夹下面的文件名，并将文件的完整路径添加到列表里
            pcd_files = self.get_folder_list(pcd_path)
            for thisfile in pcd_files:
                if thisfile.endswith(".pcd"):
                    files_list_pcd.append(str(pcd_path+'/'+thisfile))

            #获取label文件夹下面的文件名，并将文件的完整路径添加到列表里
            label_files = self.get_folder_list(label_path)
            for thisfile in label_files:
                if thisfile.endswith(".json"):
                    files_list_label.append(str(label_path +'/'+ thisfile))
        
        #返回files_list_pcd和files_list_label的列表，
        # 该列表内包含了所有pcd和label文件的路径名
        return files_list_pcd,files_list_label

    '''
    #根据一个label文件的路径single_label_path，获取该文件内的信息
    #信息包括：type, center ,size,rotation,id等信息
    def get_single_label_info(self,single_label_path):
        single_label_path = single_label_path
        #打开文件
        with open(single_label_path,encoding = 'utf-8') as f:
            labels = json.load(f)
        
        #定义一个空字典，用于存放当前帧label所有objects中的信息
        single_objects_label_info = {}
        single_objects_label_info['label_type'] = np.array([label['type'] for label in labels['labels']])
        single_objects_label_info['box_center'] = np.array([[label['center']['x'], label['center']['y'],label['center']['z']]  for  label in labels['labels']])
        single_objects_label_info['box_size'] = np.array([[label['size']['x'],label['size']['z'],label['size']['z']] for label in labels['labels']])
        single_objects_label_info['box_rotation'] = np.array([[label['rotation']['roll'],label['rotation']['pitch'],label['rotation']['yaw']]  for label in labels['labels']])
        single_objects_label_info['tracker_id'] = np.array([ label['tracker_id'] for label in labels['labels']])
        
        return single_objects_label_info
        '''

    # 根据label文件路径列表，返回所有标签的数据
    def get_all_labels(self,num_workers = 4,files_list_label=None):
        import concurrent.futures as futures

        #根据一个label文件的路径single_label_path，获取该文件内的信息
        #信息包括：type, center ,size,rotation,id等信息
        global i 
        i =0
        def get_single_label_info(single_label_path):
            global i
            i=i+1
            single_label_path = single_label_path
            #打开文件
            with open(single_label_path,encoding = 'utf-8') as f:
                labels = json.load(f)
            
            #定义一个空字典，用于存放当前帧label所有objects中的信息
            single_objects_label_info = {}
            single_objects_label_info['single_label_path'] = single_label_path
            single_objects_label_info['label_type'] = np.array([label['type'] for label in labels['labels']])
            single_objects_label_info['box_center'] = np.array([[label['center']['x'], label['center']['y'],label['center']['z']]  for  label in labels['labels']])
            single_objects_label_info['box_size'] = np.array([[label['size']['x'],label['size']['z'],label['size']['z']] for label in labels['labels']])
            single_objects_label_info['box_rotation'] = np.array([[label['rotation']['roll'],label['rotation']['pitch'],label['rotation']['yaw']]  for label in labels['labels']])
            single_objects_label_info['tracker_id'] = np.array([ label['tracker_id'] for label in labels['labels']])
            print("正在处理第 %d / %d 个数据"%(i,len(files_list_label)))
            return single_objects_label_info

        files_list_label = files_list_label
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(get_single_label_info,files_list_label)
        infos = list(infos)
        self.robosense_infos = infos
        print("*****************************Done!***********************")
        print("type  of  infos :",type(infos))
        print("len  of  infos :",len(infos))
    
        #此时的infos是一个列表，列表里面的每一个元素是一个字典，
        #每个元素里面的内容是
        return infos

    '''
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.robosense_infos) * self.total_epochs

        return len(self.robosense_infos)
        '''

    #去掉一帧里面无效的点云数据
    def remove_nan_data(self,data_numpy):
        data_numpy = data_numpy
        data_pandas = pandas.DataFrame(data_numpy)
        #删除任何包含nan的所在行 (实际有三分之一的数据无效，是[nan, nan, nan, 0.0])
        data_pandas = data_pandas.dropna(axis=0,how='any')
        data_numpy = np.array(data_pandas)

        return data_numpy

    #根据每一帧的pcd文件名和路径single_pcd_path，
    # 得到这一帧中的点云数据，返回点云的numpy格式（M,4）
    def get_single_pcd_info(self,single_pcd_path):
        single_pcd_path = single_pcd_path
        single_pcd_points = pcl.load_XYZI(single_pcd_path)
        #将点云数据转化为numpy格式
        single_pcd_points_np = single_pcd_points.to_array()
        #去掉一帧点云数据中无效的点
        single_pcd_points_np = self.remove_nan_data(single_pcd_points_np)
        #将点云数据转化为list格式
        #single_pcd_points_list =single_pcd_points.to_list()

        return single_pcd_points_np

    # 根据名字，去掉相应的信息，主要针对single_objects_label_info
    # single_objects_label_info 里关于‘unknown’的数据信息
    def drop_info_with_name(self,info,name):
        ret_info = {}
        info = info 
        keep_indices =[ i for i,x in enumerate(info['label_type']) if x != name]
        for key in info.keys():
            if key == 'single_label_path':
                continue
            ret_info[key] = info[key][keep_indices]

        return ret_info

    #根据label的路径，得到对应的pcd路径
    def from_label_path_to_pcd_path(self,single_label_path):
        #根据label的路径，推出来pcd相应的路径，两者在倒数第二个文件夹不同
        single_pcd_path = ''
        strl1 = 'label'
        strl2 = '.json'
        if strl1 in single_label_path:
            single_pcd_path = single_label_path.replace(strl1,'pcd')
        if strl2 in single_pcd_path:
            single_pcd_path = single_pcd_path.replace(strl2,'.pcd')
        #由此得到了label对应的pcd文件的路径 ：single_pcd_path
        return single_pcd_path
    
    def __getitem__(self,index):
        #if self._merge_all_iters_to_one_epoch:
        #    index = index % len(self.robosense_infos)

        single_objects_label_info = copy.deepcopy(self.robosense_infos[index])
        single_label_path = single_objects_label_info['single_label_path']
        single_pcd_path = self.from_label_path_to_pcd_path(single_label_path)

        #得到点云数据，且是有效的点云数据，返回点云的numpy格式（M,4）
        points = self.get_single_pcd_info(single_pcd_path)

        #定义输入数据的字典，包含：points，文件的路径，。。？
        input_dict = {
            'points': points,
            'frame_id': single_pcd_path,
            'single_pcd_path':single_pcd_path
        }
        
        # 在single_objects_label_info字典里，剔除关于'unknown' 的信息
        single_objects_label_info = self.drop_info_with_name(info=single_objects_label_info,name='unknown')
        label_type =single_objects_label_info['label_type']             #(N,)
        box_center = single_objects_label_info['box_center']          #(N,3)
        box_size = single_objects_label_info['box_size']                    #(N,3)
        box_rotation  = single_objects_label_info['box_rotation']  #(N,3)
        tracker_id = single_objects_label_info['tracker_id']               #(N,)
        '''
        print("——————这是去掉unknown之后的name——————")
        print(label_type)
        print(label_type.shape)
        print(box_center.shape)
        print(box_size.shape)
        print(box_rotation.shape)
        print(tracker_id.shape)
        
        print(box_rotation[:,2].reshape(-1,1))
        print(box_rotation[:,2].shape)'''

        #以下是将 上面的3D框的数据 转化为统一的数据格式
        #数据格式为：(N,7)，分别代表 (N, 7) [x, y, z, l, h, w, r]
        # gt_boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center"""
        rotation_yaw = box_rotation[:,2].reshape(-1,1)
        gt_boxes = np.concatenate([box_center,box_size,rotation_yaw],axis=1).astype(np.float32)
        print(gt_boxes.shape)
        print(type(gt_boxes))

        input_dict.update({
                'gt_names':label_type,
                'gt_boxes':gt_boxes,
                'tracker_id':tracker_id
        })
        #print(input_dict)
        
        data_dict = input_dict
        # 将点云与3D标注框均转至统一坐标定义后，送入数据基类提供的 self.prepare_data()
        #data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    
    pass

def create_robosense_infos():
    
   pass


if __name__ == '__main__':
    import sys
    from pathlib import Path
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    print(ROOT_DIR)

    
    dataset_cfg = '../../../tools/cfgs/dataset_configs/robosense_dataset.yaml'
    class_names = ['vehicle','big_vehicle','pedestrian','bicycle','cone']
    training =True
    root_path = '/root/dataset/RoboSense_Dataset/RS_datasets/datasets/'
    logger =''

    rs = RobosenseDataset(dataset_cfg= dataset_cfg,class_names=class_names,training=training, root_path=root_path,logger=logger)

    print("--------------------------这是pcd ,label的所有文件的全路径的长度--------------------")
    files_list_pcd,files_list_label =rs.get_files_name_list()
    print(len(files_list_pcd),len(files_list_label))
    
    #读取所有 label标签中 信息
    infos = rs.get_all_labels(files_list_label=files_list_label)
    #print("pcd : 前十个的信息是：")
    #print(files_list_pcd[:10])
    print("*************总信息中，第一个信息的 路径是：*****************")
    print(infos[0]['single_label_path'])

    #print("**************第五个信息得到的 标签路径、pcd路径*****************")
    data_dict=rs.__getitem__(3)
    print(data_dict)
