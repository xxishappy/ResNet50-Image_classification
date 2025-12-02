import os
import shutil
import random

source_dir = "/data/dyx/fetaldata/use_Data/test"
val_dir = "/home/dyx/MyProject/dataset/valid"
test_dir = "/home/dyx/MyProject/dataset/test"

val_ratio = 0.2  # 验证集比例，可改

# 创建目标目录
for d in [val_dir, test_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# 遍历每个标签文件夹
for label in os.listdir(source_dir):
    label_path = os.path.join(source_dir, label)
    if not os.path.isdir(label_path):
        continue
    
    # 在 val/test 中创建对应的标签目录
    os.makedirs(os.path.join(val_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    images = os.listdir(label_path)
    random.shuffle(images)
    
    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]
    test_images = images[val_count:]

    # 移动图片
    for img in val_images:
        shutil.copy(
            os.path.join(label_path, img),
            os.path.join(val_dir, label, img)
        )
    for img in test_images:
        shutil.copy(
            os.path.join(label_path, img),
            os.path.join(test_dir, label, img)
        )

print("数据划分完成！")
