import matplotlib.pyplot as plt
import numpy as np
from utils import MyDataset  # 替换为你的MyDataset所在模块名
import torchvision.transforms as transforms  # 若使用torchvision transforms



plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows默认黑体（Mac/Linux可换为'Arial Unicode MS'）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
train_dataset = MyDataset(flag='train')
valid_dataset = MyDataset(flag='valid')
test_dataset = MyDataset(flag='test')


plt.figure(figsize=(10, 6))

# 统计各数据集样本数
dataset_names = ['训练集', '验证集', '测试集']
sample_counts = [len(train_dataset), len(valid_dataset), len(test_dataset)]

# 设置柱状图样式
colors = ['#2E86AB', '#A23B72', '#F18F01']
bars = plt.bar(dataset_names, sample_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for bar, count in zip(bars, sample_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{count:,}',  # 千分位格式化
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# 设置标题和标签
plt.title('三个数据集样本数量对比', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('样本数量', fontsize=14)
plt.xlabel('数据集', fontsize=14)

# 调整y轴范围（避免数值紧贴顶部）
plt.ylim(0, max(sample_counts) * 1.1)

# 添加网格线（仅y轴，增强可读性）
plt.grid(axis='y', alpha=0.3, linestyle='--')

# 去除顶部和右侧边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('dataset_sample_count.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------- 4. 可视化2：数据集类别分布柱状图（分类任务必备）----------------------
# 前提：MyDataset的__getitem__返回 (data, label)，label为单类别整数（0,1,2...）
def count_class_distribution(dataset):
    """统计数据集的类别分布"""
    class_counts = {}
    for _, label,_ in dataset:
        label = int(label)  # 确保label是整数
        class_counts[label] = class_counts.get(label, 0) + 1
    # 按类别排序
    sorted_classes = sorted(class_counts.keys())
    sorted_counts = [class_counts[c] for c in sorted_classes]
    return sorted_classes, sorted_counts

# 统计三个数据集的类别分布
train_classes, train_class_counts = count_class_distribution(train_dataset)
valid_classes, valid_class_counts = count_class_distribution(valid_dataset)
test_classes, test_class_counts = count_class_distribution(test_dataset)

# 确保所有数据集类别一致（取并集）
all_classes = sorted(list(set(train_classes + valid_classes + test_classes)))
class_names = [f'类别{c}' for c in all_classes]

# 统一各类别的计数（不存在的类别计数为0）
def uniform_class_counts(classes, counts, all_classes):
    uniform_counts = []
    for c in all_classes:
        if c in classes:
            uniform_counts.append(counts[classes.index(c)])
        else:
            uniform_counts.append(0)
    return uniform_counts

train_uniform = uniform_class_counts(train_classes, train_class_counts, all_classes)
valid_uniform = uniform_class_counts(valid_classes, valid_class_counts, all_classes)
test_uniform = uniform_class_counts(test_classes, test_class_counts, all_classes)

# 绘制分组柱状图
plt.figure(figsize=(12, 7))
x = np.arange(len(all_classes))  # 类别位置
width = 0.25  # 柱子宽度

# 绘制三组柱子
bars1 = plt.bar(x - width, train_uniform, width, label='训练集', color='#2E86AB', alpha=0.8, edgecolor='black')
bars2 = plt.bar(x, valid_uniform, width, label='验证集', color='#A23B72', alpha=0.8, edgecolor='black')
bars3 = plt.bar(x + width, test_uniform, width, label='测试集', color='#F18F01', alpha=0.8, edgecolor='black')

# 添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # 只给非零值加标签
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# 设置标题和标签
plt.title('三个数据集的类别分布对比', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('样本数量', fontsize=14)
plt.xlabel('类别', fontsize=14)
plt.xticks(x, class_names, fontsize=12)
plt.legend(fontsize=12)

# 添加网格线
plt.grid(axis='y', alpha=0.3, linestyle='--')

# 去除顶部和右侧边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('dataset_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()