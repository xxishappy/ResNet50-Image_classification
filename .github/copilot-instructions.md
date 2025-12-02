# Copilot 指令（为 AI 代码助手量身定制）

目的：让 AI 代理快速上手此代码库，理解主要架构、数据流、开发/运行工作流以及常见约定，给出可复制的修改和运行示例。

要点摘要
- 项目类型：PyTorch 图像分类（ResNet50）训练/评估脚本（单机脚本式，不是大型工程化服务）。
- 主要脚本：`train_Res.py`（训练）、`test_res.py`（测试/评估）、`datapre.py`（Dataset 定义）、`datavis.py`（可视化辅助）。
- 运行假设：代码中很多数据路径为绝对路径，运行前须更新为当前环境的路径或设置软链接。

大体架构（短）
- 数据读取：`datapre.train_v_Dataset` 读取训练/验证数据，期望 `img_dir` 下按数字命名的子文件夹（每个子文件夹名为类索引），每个子文件夹内为图片文件；`__getitem__` 返回 `(image, label, img_path)`。
- 测试读取：`datapre.test_Dataset` 使用一个 JSON（`data['test']`）来映射 `(filename, label, uid)`，其 `__getitem__` 返回 `(image, label, uid, img_path)`。
- 训练脚本：`train_Res.py` 使用 torchvision 的 `resnet50` + 自定义 classifier（2048→500→num_classes），默认冻结 backbone 参数，仅训练 `model.fc`。
- 日志与监控：使用 `wandb.init(project="ResNet50", name="exp6")`，同时保存训练日志 `training_log.txt` 与最优模型至 `./ResNet/weights/resnet50.pth`。

关键约定与易错点（请严格遵守）
- 数据目录约定：训练/验证目录每个子目录名必须能被 `int()` 转换（例如 `0`, `1`, ...）；代码通过 `sorted(os.listdir(img_dir), key=int)` 保证标签顺序。
- 图片预处理：使用 `pre_img(image, crop_size=224)` 统一缩放并填充到 224×224，再做 ToTensor+Normalize（mean/std 已硬编码为 ImageNet 值）。
- Batch/并行：`DataLoader` 在训练中使用 `num_workers=8`，在低核或 Windows 环境请降低或设为 0。
- WandB：脚本直接调用 `wandb.init`，所以在无登录时会报错或打开交互页面。建议在 CI/本地运行前设置 `WANDB_API_KEY` 或删除/注释掉 `wandb` 调用以避免阻塞。

运行与调试（可复制命令）
- 训练（默认直接用文件顶部的绝对路径，建议先修改为本地路径）：
```
python train_Res.py
```
- 测试/绘图：
```
python test_res.py
```
- 如果要在新环境运行：
  - 创建虚拟环境并安装常用依赖：`torch torchvision scikit-learn matplotlib tqdm wandb pillow seaborn`。
  - 更新脚本顶部的 `train_root`/`valid_root`/`test_root`/`json_root` 为项目数据路径。

代码库里发现的可改进/注意事项（供 AI 直接修复时参考）
- `test_res.py` 中的 `get_predictions` 函数引用了未定义的 `max_batches` 与 `batch_idx`，且函数签名在被调用时实参不匹配（被调用为 `get_predictions(model,test_loader,max_batches=2)`）。建议：
  - 明确 `get_predictions(model, iterator, max_batches=None)`，在循环内使用 `for batch_idx, (x,y,_,_) in enumerate(iterator):` 并在开始处检查 `if max_batches is not None and batch_idx>=max_batches: break`。
- 模型加载路径是硬编码绝对路径（例如 `torch.load("/home/dyx/.../resnet50.pth")`），AI 修改时请把路径提成变量或使用 CLI 参数/环境变量。
- `train_Res.py` 在保存最优模型时用 `torch.save(model.state_dict(), best_model_path)`，测试脚本用 `model.load_state_dict(torch.load(..., map_location=device))` — 保持一致即可。

示例：修复 `test_res.get_predictions` 的建议实现
```
def get_predictions(model, iterator, max_batches=None):
    model.eval()
    images, labels, probs = [], [], []
    with torch.no_grad():
        for batch_idx, (x,y,_,_) in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device)
            y_pred = model(x)
            y_prob = torch.exp(y_pred)
            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())
    return torch.cat(images), torch.cat(labels), torch.cat(probs)
```

如何让 AI 继续贡献（交互指引）
- 如果想修改训练超参：在 `train_Res.py` 顶部查找并修改 `batch_size`, `epochs`, `lr`, `weight_decay` 等变量。
- 如果要增加数据增强或换用预训练权重：查看 `transform_train` / `transform_valid` 以及 `weights = ResNet50_Weights.DEFAULT` 的使用位置。
- 如需替换模型骨架或解冻 backbone：在加载 `resnet50` 后调整 `for param in model.parameters(): param.requires_grad = False` 的逻辑，或设置 `requires_grad=True`。

反馈请求
- 我已基于可见源码提取关键流程。请确认：
  - 数据是否的确按 `img_dir/<class_index>/<image>` 布局存放？
  - 是否希望我把 `train_root` 等硬编码路径改为相对路径或 CLI 参数？
  - 是否需要我把 WandB 初始化改为可选（通过环境变量控制）？

如需我现在直接修复 `test_res.py` 的 `get_predictions`，或把训练/测试脚本改为使用命令行参数，请告知，我会继续实现并运行小范围自检。
