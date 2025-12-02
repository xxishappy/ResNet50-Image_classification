import os 
import json
import glob
from torch.utils.data import Dataset
from PIL import Image

#Dataset:提供一种方式来获取数据及其标签
#构建训练集和验证集的数据读取
class train_v_Dataset(Dataset):
    def __init__(self,file_path, root, flag='train', transform=None):
        '''
        img_dir:图像数据集文件夹路径
        每个子文件夹名对应图像的label
        '''
        self.img_file = json.load(open(file_path, 'r', encoding='utf-8'))
        img_dict = self.img_file[flag]  #获取训练集或验证集的图像信息列表

        self.samples=[]  #存放所有图片路径及其标签的列表
        for img_name, infor in img_dict.items():
            label = infor[0]
            img_path = f"{root}/{flag}/{label}/{img_name}"
            self.samples.append((img_path,label))   # eg.("train/0/a.png", 0)'''

        self.transform=transform

    def __len__(self):
        return len(self.samples) #返回样本数量

    def __getitem__(self,idx):
        img_path,label=self.samples[idx]
        label = int(label)
        image=Image.open(img_path)   #Image.open()返回的是PIL.Image对象，方便做图像处理
        if self.transform:
            image = self.transform(image)
        return image,label,img_path
    
    
# #构建测试集的数据读取
# class test_Dataset(Dataset):
#     def __init__(self,img_dir,json_root,transform=None,target_transform=None):
#         '''
#         img_dir:图像数据集文件夹路径
#         json_root:文件名---标签的json文件路径
#         '''
#         self.img_dir=img_dir

#         #获取标签
#         #读取json文件，获取标签
#         with open(json_root,'r',encoding='utf-8') as f:
#             data=json.load(f)  #解析json文件中的内容，将json数据转换为python对象
        
#         self.samples=[]
#         for i in data['test']:
#             filename=i[0]
#             label=int(i[1])
#             uid=i[2]
#             img_path=os.path.join(img_dir,uid,filename)
#             self.samples.append((img_path,label,uid))

#         self.transform=transform
#         self.target_transform=target_transform
        
#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self,idx):
#         img_path,label,uid=self.samples[idx]
#         image=Image.open(img_path)
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image,label,uid,img_path

