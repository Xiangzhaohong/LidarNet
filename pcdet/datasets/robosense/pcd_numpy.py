s = 'abcdf/sdf/label/kkk.json'
strl1 = 'label'
strl2 ='.json'
if strl1 in s :
    s = s.replace(strl1, 'pcd')
if strl2 in s:
    s=s.replace(strl2,'.pcd')
print (s)

'''
import string
strsrc = 'abc'
notrans = string.maketrans('','') #替换创建
print (strsrc.translate(notrans)) 
trans12 = string.maketrans('ab','12') #替换创建
print (strsrc.translate(trans12)) 
'''


'''
import pcl
import numpy as np
import pandas

root_path = '/root/dataset/RoboSense_Dataset/RS_datasets/datasets/ruby_ruby136_shizilukou_1200526171538/pcd/'
#p = pcl.load(root_path+'ruby_ruby136_shizilukou_1200526171538_15.pcd')
p = pcl.load_XYZI(root_path+'ruby_ruby136_shizilukou_1200526171538_15.pcd')
#p = pcl.load_XYZRGB(root_path+'ruby_ruby136_shizilukou_1200526171538_15.pcd')


print(p)
print(type(p))
#fil  = p.make_statistical_oulier_filter()
print('p[0]:',p[0])
print("p[1]:",p[1])


m = p.to_array()    #注意，里面有点数据是nan，需要剔除！
print(m)
print(type(m))
print(m.shape)
print("--------------------去掉NAN--------------------")

m2 = pandas.DataFrame(m)
m2 = m2.dropna(axis=0,how='any')
print(m2)
print(type(m2))
print(m2.shape)
print("---------------------------------------")


n = p.to_list()
print(type(n))
num =0
for i in n:
    if str(i) =='[nan, nan, nan, 0.0]':
        num = num+1
print(num)
#print(n)
#print(len(list))
print("完成！！！！！！！！！！")
'''
