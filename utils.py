from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, flag='train', transform=None):
        """
        flag: 'train' / 'valid' / 'test'
        """

        root="/data/dyx/fetaldata/48dataset"
        self.flag = flag
        self.transform = transform
        self.samples = []

        flag_dir = os.path.join(root, flag)

        if not os.path.exists(flag_dir):
            raise FileNotFoundError(f"路径不存在：{flag_dir}")

        # 遍历 flag 目录下的子文件夹（标签）
        for label_name in os.listdir(flag_dir):
            label_path = os.path.join(flag_dir, label_name)
            if not os.path.isdir(label_path):
                continue

            # 子文件夹名作为标签
            try:
                label = int(label_name)
            except:
                raise ValueError(f"子文件夹名 {label_name} 无法转换为整数标签")

            # 遍历图片文件
            for img_name in os.listdir(label_path):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    img_path = os.path.join(label_path, img_name)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  #ResNet需要3通道图像

        if self.transform:
            image = self.transform(image)

        return image, label, img_path
