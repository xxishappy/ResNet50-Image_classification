from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image,ImageFile
from torchvision import models
import torch
from  tqdm import tqdm
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score,confusion_matrix
import matplotlib.pyplot as plt
from utils import MyDataset
import seaborn as sns
from collections import defaultdict
import json
import os
import csv

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pre_img(image, crop_size=224):
    width, height = image.size
    scale = crop_size / max(width, height)
    new_w = int(width * scale)
    new_h = int(height * scale)
    image_resized = image.resize((new_w, new_h), Image.BICUBIC)
    new_image = Image.new(image.mode, (crop_size, crop_size), (0, 0, 0))
    left = (crop_size - new_w) // 2
    top = (crop_size - new_h) // 2
    new_image.paste(image_resized, (left, top))
    return new_image


transform = transforms.Compose([
    transforms.Lambda(lambda img: pre_img(img, crop_size=224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = MyDataset( flag='test', transform=transform) 
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)


device=torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
# 2. 获取原fc层的输入特征维度（关键！不同ResNet版本输入维度不同）
in_features = model.fc.in_features  # ResNet18/34: 512; ResNet50/101/152: 2048
# 3. 替换为自定义fc层（num_classes为你的类别数，比如10类）
num_classes = 48
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),   #增加Dropout层，防止过拟合
    nn.Linear(in_features, num_classes))

state_dict = torch.load("/home/dyx/MyProject/weights/exp4/resnet50_best_model_0.6757110357284546.pth", map_location='cpu')
model.load_state_dict(state_dict)
model=model.to(device)
model.eval()    

num="2"
save_dir = "./test/" + num
os.makedirs(save_dir, exist_ok=True)
#按uid分组统计
with open("cls_test_std_data_with_cls_id.json","r",encoding="utf-8") as f:
    test_json=json.load(f)

#文件名与uid对应
filename_to_uid = {}
for item in test_json["test"]:
    filename=item[0]
    label=item[1]
    uid=item[2]
    root="/data/dyx/fetaldata/48dataset/test"
    if not os.path.exists(os.path.join(root,str(label),filename)):
        continue  #因为原测试集分为了现在的验证集和测试集，部分文件不存在，跳过
    filename_to_uid[filename]=uid

test_preds=[]
test_labels=[]
test_uids=[]
uid_labels=defaultdict(list)
uid_preds=defaultdict(list)

with torch.no_grad():
    for images,labels,img_paths in tqdm(test_loader):
        images=images.to(device)
        labels=labels.to(device)
        

        outputs=model(images)
        _, predicted = outputs.max(1)

        for fname,label,pred in zip(img_paths,labels.cpu().numpy(),predicted.cpu().numpy()):
            basename=os.path.basename(fname) #获取文件名
            uid=filename_to_uid.get(basename,None)
            if uid is  None:
                continue
            
            test_labels.append(label)
            test_preds.append(pred)
            test_uids.append(uid)

            uid_labels[uid].append(label)
            uid_preds[uid].append(pred) 


all_accuracy = accuracy_score(test_labels, test_preds)
all_precision = precision_score(test_labels, test_preds, average='macro')
all_recall = recall_score(test_labels, test_preds, average='macro',zero_division=0)
all_f1 = f1_score(test_labels, test_preds, average='macro')

#保存记录

csv_file=os.path.join(save_dir,"test_results.csv") 
with open(csv_file, "w", encoding="utf-8") as f:
    writer=csv.writer(f) 
    #总指标
    writer.writerow(["总体指标"])
    writer.writerow(["Accuracy","Precision","Recall","F1 Score"])
    writer.writerow([f"{all_accuracy:.4f}",f"{all_precision:.4f}",f"{all_recall:.4f}",f"{all_f1:.4f}"])
    writer.writerow([])
    #按uid分组指标
    writer.writerow(["按UID分组指标"])
    # writer.writerow(["UID","Accuracy","Precision","Recall","F1 Score","样本数"])
    writer.writerow(["UID","Accuracy","样本数"])
    for uid in uid_labels:
        y_true=uid_labels[uid]
        y_pred=uid_preds[uid]
        acc=accuracy_score(y_true,y_pred)
        # prec=precision_score(y_true,y_pred,average="weighted",zero_division=0)
        # rec=recall_score(y_true,y_pred,average="weighted",zero_division=0)
        # f1= f1_score(y_true,y_pred,average="weighted",zero_division=0)
        num_samples=len(y_true)
        # writer.writerow([uid,f"{acc:.4f}",f"{prec:.4f}",f"{rec:.4f}",f"{f1:.4f}",num_samples])
        writer.writerow([uid,f"{acc:.4f}",num_samples])

print(f"\n测试结果已保存到{csv_file}中") 
        
#绘制混淆矩阵
cm = confusion_matrix(test_labels, test_preds)
#可视化混淆矩阵
plt.figure(figsize=(20,16))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues') #annot=False 避免太混乱
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.figtext(0.5, 0.01, f'Accuracy: {all_accuracy:.4f}, Precision: {all_precision:.4f}, Recall: {all_recall:.4f}, F1 Score: {all_f1:.4f}', ha='center', fontsize=12)
plt.savefig(os.path.join(save_dir,'confusion_matrix.png'))
plt.close()


