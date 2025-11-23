# 2025 VLP挑战赛参赛作品 - 路面裂缝语义分割

<div align="center">

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-MMSegmentation-green)](https://github.com/open-mmlab/mmsegmentation)
[![Rank](https://img.shields.io/badge/Rank-5th-yellow)](README.md)

</div>

## 1. 项目简介 (Introduction)

本项目是 **2025 VLP挑战赛** (2025 VLP Challenge) 的参赛代码仓库。

针对路面裂缝检测任务中裂缝细微、背景复杂等难点，我们基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 框架进行了深度优化。

### 核心亮点
* **模型架构**：以 **Segformer-b1** 为基座模型，平衡了推理速度与分割精度。
* **注意力机制**：引入 **SimAM (Simple Attention Module)** 无参注意力机制，增强了模型对细微裂缝特征的提取能力。
* **混合损失函数**：构建了一个混合损失函数 (**CrossEntropyLoss + LovaszLoss**)。
    * CrossEntropyLoss（权重 1.0）作为主损失；
    * LovaszLoss（权重 1.0）作为辅助损失，帮助模型更好地理解裂缝的“细长空间关系”，从而生成更完整、更连续的预测掩膜。
* **模型融合**：采用了多模型加权融合策略 (Model Fusion)，显著提升了最终的 IoU 分数。
* **当前成绩**：在榜单中暂列 **第 5 名**。

---

## 2. 环境配置 (Installation)

本项目依赖 Python 3.8+ 和 PyTorch 1.10+。建议使用 Anaconda 进行环境管理。

### 2.1 基础环境安装
```bash
# 1. 克隆本仓库
git clone https://github.com/hanghang16/UVA-CrackSeg.git
cd UVA-CrackSeg

# 2. 创建虚拟环境
conda create -n vlp_seg python=3.9 -y
conda activate vlp_seg

# 3. 安装 PyTorch (根据你的 CUDA 版本调整，以下以 CUDA 11.3 为例)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 4. 安装 MMCV (使用 openmim 安装以避免版本冲突)
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

### 2.2 安装项目依赖

```bash
# 安装本项目及其他依赖库
pip install -v -e .
pip install -r requirements.txt
```

-----

## 3\. 数据集准备 (Data Preparation)

请确保数据集按照以下结构放置在 `data/` 目录下：

```text
UVA-CrackSeg/
├── configs/
├── tools/
├── ...
└── data/
    └── vlp_dataset/          # [请修改为你实际的数据集文件夹名]
        ├── images/
        │   ├── train/
        │   └── val/
        └── annotations/
            ├── train/
            └── val/
```

-----

## 4\. 运行说明 (Usage)

### 4.1 训练 (Training)

使用单卡进行训练：

```bash
# 训练 Segformer-b1 + SimAM 模型
python tools/train.py Crack-Configs/你的配置文件名.py
```

### 4.2 测试与评估 (Inference & Evaluation)

加载训练好的权重文件进行测试，并计算 mIoU：

```bash
python tools/test.py \
    Crack-Configs/你的配置文件名.py \
    work_dirs/你的权重文件.pth \
    --eval mIoU
```

-----

## 5\. 开源许可证 (License)

本项目遵循 **Apache 2.0 License** 开源协议，与 MMSegmentation 保持一致。详细条款请参见 [LICENSE](https://github.com/hanghang16/UVA-CrackSeg/blob/main/LICENSE) 文件。

-----

## 6\. 致谢 (Acknowledgement)

本项目基于优秀的开源语义分割框架 **MMSegmentation** 开发。感谢 OpenMMLab 团队的贡献。

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```
