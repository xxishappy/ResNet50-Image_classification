import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
# 总映射表
total_labels = {
    0: "背景",
    1: "脊柱矢状切面全段",
    2: "脊髓圆锥定位切面",
    3: "丘脑水平横切面",
    4: "侧脑室水平横切面",
    5: "小脑水平横切面",
    6: "上牙槽突切面",
    7: "双眼球水平横切面（晶状体）",
    8: "颜面部正中矢状面",
    9: "唇冠状切面",
    10: "四腔心切面",
    11: "四腔心切面（带血流）",
    12: "主动脉弓长轴切面",
    13: "主动脉弓长轴切面（带血流）",
    14: "右心室流出道切面 ",
    15: "三血管气管切面",
    16: "三血管气管切面（带血流）",
    17:"左心室流出道切面",
    18:"导管弓切面",
    19:"导管弓切面（带血流）",
    20:"肺动脉分叉切面",
    21: "上腹部横切面（腹围切面）",
    22:"胆囊切面",
    23:"双肾横切面",
    24:"左肾纵切面",
    25: "脐动脉水平膀胱横切面",
    26:"肱骨长轴切面",
    27: "上肢切面",
    28:"手切面",
    29: "下肢切面",
    30:"股骨长轴切面",
    31:"足切面",
    32: "孕妇宫颈管矢状切面",
    33: "脐带腹壁入口腹部横切面",
    34: "脐带胎盘插入口",
    35:"右肾纵切面",
    36:"膈肌切面",
    37:"上下腔静脉长轴切面",
    38:"肾冠状切面",
    39:"肾动脉冠状切面（带血流）",
    40:"左心室流出道切面（带血流）",
    41:"右心室流出道切面（带血流）",
    42:"肺动脉分叉切面（带血流）",
    43:"硬腭切面",
    44:"软腭切面",
    45: "胎儿正中矢状切面",
    46: "NT测量切面",
    47: "颅脑横切面",
}

def count_images_in_folder(dataset_dir):
    counts = {}
    for label_name in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_name)
        if os.path.isdir(label_path):
            # 统计图像文件数量
            num_imgs = len([f for f in os.listdir(label_path)
                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
            counts[int(label_name)] = num_imgs
    return counts
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False   
def plot_and_save(counts, title, save_path):
    sorted_ids = sorted(counts.keys())
    class_names = [total_labels.get(c, str(c)) for c in sorted_ids]
    img_counts = [counts[c] for c in sorted_ids]
    
    plt.figure(figsize=(12,9))
    bars = plt.bar(class_names, img_counts, color='skyblue')
    plt.xlabel('类别')
    plt.ylabel('数量')
    plt.title(title)
    plt.xticks(rotation=90, ha='center')
    
    # 在柱状图顶端显示数量
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, str(height), 
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 根目录
root = "/data/dyx/fetaldata/48dataset"

# 统计 train/valid/test
train_counts = count_images_in_folder(os.path.join(root, "train"))
valid_counts = count_images_in_folder(os.path.join(root, "valid"))
test_counts  = count_images_in_folder(os.path.join(root, "test"))

# 绘图并保存
plot_and_save(train_counts, "训练集类别分布", "./fig/train_distribution.png")
plot_and_save(valid_counts, "验证集类别分布", "./fig/valid_distribution.png")
plot_and_save(test_counts, "测试集类别分布", "./fig/test_distribution.png")

def print_total(counts, dataset_name):
    total = sum(counts.values())
    print(f"{dataset_name} 总图片数量: {total}")

print_total(train_counts, "训练集")
print_total(valid_counts, "验证集")
print_total(test_counts, "测试集")