# encoding=utf-8
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from utils import MyDataset
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

##1.数据准备部分
#自定义图像预处理，进行缩放-填充-裁剪
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

# 新增噪声增强类 
class AddSpeckleNoise(object):
    def __init__(self, std=0.1):
        self.std = std
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return tensor + tensor * noise

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    
#定义transform
transform_train = transforms.Compose([
    transforms.Lambda(lambda img: pre_img(img, crop_size=224)), 
    transforms.RandomHorizontalFlip(p=0.5),        # 水平翻转
    transforms.RandomRotation(10),          # 轻微旋转
    transforms.ColorJitter(brightness=0.1,   # 轻微亮度
                           contrast=0.1,     # 轻微对比度
                           saturation=0, #不改变饱和度
                           hue=0),       #不改变色相                          
    transforms.RandomAffine(degrees=0, translate=(0.05,0.05)),  # 轻微平移
    # 转 tensor + 标准化
    transforms.ToTensor(),
    AddSpeckleNoise(std=0.1),   # 添加高斯噪声
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_valid = transforms.Compose([
    transforms.Lambda(lambda img: pre_img(img, crop_size=224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 数据集和数据加载器
train_dataset = MyDataset( flag='train', transform=transform_train)
valid_dataset = MyDataset( flag='valid', transform=transform_valid) 

train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)
#pin_memory=True 会让DataLoader将数据加载到固定内存(page-locked memory)中，这样可以加快数据从CPU传输到GPU的速度，提升训练效率。
#drop_last=True 如果数据集大小不能被批次大小整除，丢弃最后一个不完整的批次。


##2.模型训练部分
num="4"
exp = 'exp'+num
#WandB 初始化 
wandb.init(project="ResNet50", name=exp)

#模型加载 
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
# 获取原fc层的输入特征维度（关键！不同ResNet版本输入维度不同）
in_features = model.fc.in_features  # ResNet18/34: 512; ResNet50/101/152: 2048
# 替换为自定义fc层
num_classes = 48
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),   #增加Dropout层，防止过拟合
    nn.Linear(in_features, num_classes))
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #label_smoothing=0.1 平滑标签，防止过拟合
optimizer = optim.AdamW(model.parameters(), lr=1e-4,weight_decay=1e-4)

#绝命错误
# optimizer = optim.AdamW(model.fc.parameters(), lr=3e-4,weight_decay=1e-4)
model = model.to(device)

#记录准备
save_dir = "./weights/" + exp
os.makedirs(save_dir, exist_ok=True)
log_file = "./training_log.txt"
history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [], "val_acc": [],
    "train_f1": [], "val_f1": []
}

epochs = 200
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
#ReduceLROnPlateau 会根据验证集指标自动调整学习率  factor=0.5:表示每次降低学习率时，将当前学习率乘以0.5  patience=5:表示如果经过5个epoch验证集指标没有提升，就降低学习率

#增加早停机制
early_stopping_patience = 20
no_improve=0
best_val_f1 = 0.0

# -------------------- 训练循环 --------------------
for epoch in range(epochs):
    print(f"--------第{epoch+1}轮训练开始-------")
    model.train()

    train_acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    train_f1_metric = MulticlassF1Score(num_classes=num_classes,average="macro").to(device)

    train_loss = 0.0

    for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", ncols=100):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 梯度清零
        outputs = model(images)# 前向传播，即根据当前参数做一次推理
        loss = criterion(outputs, labels) # 计算损失
        loss.backward() # 反向传播，找出怎么改正错误
        optimizer.step()# 用梯度更新参数
        #前向传播（算答案） → 计算损失（算错到什么程度） → 反向传播（找原因） → 参数更新（改进自己）

        train_loss += loss.item() * images.size(0) # 累加损失
        _, predicted = outputs.max(1) # 获取预测结果
        
        #更新指标
        train_acc_metric.update(predicted,labels)
        train_f1_metric.update(predicted,labels)

    #epoch 结束后计算平均损失和指标
    train_loss = train_loss / len(train_loader.dataset)
    train_acc=train_acc_metric.compute().item() 
    train_f1=train_f1_metric.compute().item()


    ##3.模型验证部分
    model.eval()
    val_loss = 0.0
    val_acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    val_f1_metric = MulticlassF1Score(num_classes=num_classes,average="macro").to(device)

    with torch.no_grad():
        for images, labels, _ in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Valid]", ncols=100):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)

            #更新指标
            val_acc_metric.update(predicted,labels)
            val_f1_metric.update(predicted,labels)
            

    #epoch 结束后计算平均损失和指标
    val_loss = val_loss / len(valid_loader.dataset)
    val_acc=val_acc_metric.compute().item() 
    val_f1=val_f1_metric.compute().item()

    # 日志记录
    with open(log_file, "a") as f:
        f.write(f"start_time:{datetime.datetime.now()}\n")
        f.write(
            f"epoch:{epoch+1},train_loss:{train_loss:.4f},val_loss:{val_loss:.4f},"
            f"train_acc:{train_acc:.4f},val_acc:{val_acc:.4f},"
            f"train_f1:{train_f1:.4f},val_f1:{val_f1:.4f},"
            f"end_time:{datetime.datetime.now()}\n")
        f.flush()

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1
    })

    # -------------------- 保存最优模型 --------------------
    if val_f1 > best_val_f1: #判断当前验证 F1 是否为最佳
        best_val_f1 = val_f1
        no_improve=0
        torch.save(model.state_dict(), f"{save_dir}/resnet50_best_model_{best_val_f1}.pth")
        print(f"验证 F1 提升至 {best_val_f1:.4f}，已保存最优模型到: {save_dir}/resnet50_best_model_{best_val_f1:.4f}.pth")
    else:
        no_improve += 1
        print(f"验证 F1 未提升，当前连续未提升轮数: {no_improve}/{early_stopping_patience}")
        if no_improve >= early_stopping_patience:
            print("验证 F1 已连续多轮未提升，触发早停机制，结束训练。")
            break

    # -------------------- 更新 history --------------------
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)
    scheduler.step(val_f1)

# 4.绘制训练曲线图
epochs_range = range(1, len(history["train_loss"]) + 1)
plt.figure(figsize=(14, 6))

# Loss 曲线
plt.subplot(1, 3, 1)
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy 曲线
plt.subplot(1, 3, 2)
plt.plot(epochs_range, history["train_acc"], label="Train Acc")
plt.plot(epochs_range, history["val_acc"], label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

# F1 曲线
plt.subplot(1, 3, 3)
plt.plot(epochs_range, history["train_f1"], label="Train F1")
plt.plot(epochs_range, history["val_f1"], label="Val F1")
plt.title("F1 Score Curve")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves"+num+".png", dpi=300)
plt.close()


'''
1.训练模型要看模型架构；
2.检查代码要看训练集和验证集在训练中的结果与在测试时的结果是否一致；
'''