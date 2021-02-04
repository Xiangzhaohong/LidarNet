
def testkkk():
	print ("testkkk测试用成功 ")


'''

# coding: utf-8
import json
import numpy as np

root_path = '/root/dataset/RoboSense_Dataset/RS_datasets/datasets'
with open(root_path+'/ruby119_longzhudadao_1200423181920/label/ruby119_longzhudadao_1200423181920_770.json', encoding='utf-8') as f:
	labels = json.load(f)
i=0
single_objects_label_info = {}
single_objects_label_info['label_type'] = np.array([label['type'] for label in labels['labels']])



#print(single_objects_label_info['label_type'])
#print("-----------------",type(single_objects_label_info['label_type']))

single_objects_label_info['box_center'] = np.array([[label['center']['x'], label['center']['y'],label['center']['z']]  for  label in labels['labels']])
#print(single_objects_label_info['box_center'] )
#print (type(single_objects_label_info['box_center']))

single_objects_label_info['box_size'] = np.array([[label['size']['x'],label['size']['z'],label['size']['z']] for label in labels['labels']])
#print(single_objects_label_info['box_size'] )
print (type(single_objects_label_info['box_size']))
print(single_objects_label_info['box_size'].shape)

single_objects_label_info['box_rotation'] = np.array([[label['rotation']['roll'],label['rotation']['pitch'],label['rotation']['yaw']]  for label in labels['labels']])
#print(single_objects_label_info['box_rotation'])
print (type(single_objects_label_info['box_rotation']))
print(single_objects_label_info['box_rotation'].shape)

single_objects_label_info['tracker_id'] = np.array([ label['tracker_id'] for label in labels['labels']])
print(single_objects_label_info['tracker_id'])
print (type(single_objects_label_info['tracker_id']))
print(single_objects_label_info['tracker_id'].shape)

print (single_objects_label_info)

'''

'''
for label in labels['labels']:
    # 获取3d标注框基本信息
	i+=1
	print('第%d个物体：'%(i))
	label_type = label['type']  # str: 标注框类别
	print('type:',label_type)
	box_center = label['center'] # dict: 包含x, y, z
	print('box_center:',box_center)
	box_size =  label['size']  # dict: 包含x, y, z
	print('box_size:',box_size)
	box_rotation = label['rotation']  # dict: 包含roll, pitch, yaw
	print('box_rotation:',box_rotation)
	tracker_id = label['tracker_id']  # int: 标注框trackerid
	print('tracker_id:',tracker_id)
	print('')


print('一共有 %d 个物体'%i)
	'''
	



