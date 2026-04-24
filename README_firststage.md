# LightenDiffusion - 第一阶段训练

## 实现依据说明

本目录下的第一阶段训练代码由我们根据论文内容和开源代码**补充实现**，原始仓库未提供。以下区分论文明确给出的信息与我们的推测/补充部分：

### 论文明确提供的信息

- **网络结构**：CTDN 的完整架构（编码器 feature_pyramid、Retinex_decom 含 Cross/Self Attention、解码器 ReconNet），对应 `models/decom.py`，由作者开源
- **损失函数定义**：L_con（Eq.7）、L_rec（Eq.8）、L_ref（Eq.9）、L_ill（Eq.9）的公式
- **损失权重**：λ_1=0.01、λ_ref=0.1、λ_ill=0.01、λ_g=10（论文 Section 4.1）
- **训练数据来源**：使用 SICE 数据集的配对低质量图像（论文 Section 3.4："we follow [10] that utilizes paired low-quality images from the SICE dataset [3]"）
- **优化器与学习率**：Adam，初始 lr=1e-4，衰减系数 0.8（论文 Section 4.1）
- **总迭代次数**：1×10^5 次（论文 Section 4.1）
- **特征下采样级数**：k=3（论文 Section 4.1）
- **权重保存格式**：由第二阶段 `load_stage1()` 的代码反推确定，格式为 `{'model': state_dict}`
- **第二阶段冻结 CTDN**：由 `ddm.py` 中 `requires_grad = False` 逻辑确认

### 推测或补充的部分

- **SICE 数据集的配对方式**：论文仅说"paired low-quality images from SICE"，具体如何从多曝光序列中组成配对（随机选取两张不同曝光？固定选最暗和最亮？）**未明确说明**，我们实现为任意两张不同曝光图像配对
- **L_con 的具体实现路径**：论文给出公式 `||I - D(E(I))||_2`，但编码器产出的潜在特征如何经过解码器重建回原图的数据流（是否使用 skip connection、是否用 `channel_up`），是根据 `ReconNet` 代码中 `pred_fea` 分支的逻辑推断的
- **λ_1 的用途**：论文列出 4 个超参数 λ_1=0.01、λ_2=0.1、λ_3=0.01、λ_g=10，但仅在 L_dec 公式中使用了 λ_2、λ_3、λ_g。λ_1 未在任何公式中显式出现，我们推断其作为 L_con 在总损失中的权重：`L = λ_1 · L_con + L_dec`
- **学习率衰减策略**：论文说"decays by a factor of 0.8"，但未说明衰减间隔和方式（StepLR？每 N 步？），我们设定为每 20000 步衰减一次
- **batch size 与显存约束**：论文提到 batch_size=12，但未区分阶段。第一阶段显存需求远高于第二阶段（第二阶段冻结 CTDN，不保存激活），具体分析：
  > **显存估算**：`feature_pyramid` 在 512×512 下每次调用约保存 7 个 `[B,64,512,512]` 级张量用于反向传播，单次约 B×750MB。第一阶段 `compute_losses` 共 4 次 pyramid 调用（编码 2 次 + 解码跳跃连接 2 次），激活显存约 4×B×750MB。  
  > - batch=12: 约 36GB 激活 + 2GB 其他 ≈ **38GB**（需 48GB+ GPU）  
  > - batch=4: 约 12GB 激活 + 2GB 其他 ≈ **14GB**（适合 16GB GPU）  
  > - batch=2: 约 6GB 激活 + 2GB 其他 ≈ **8GB**（适合 12GB GPU）  
  >
  > 我们默认设为 batch=4，适配 16GB 显存。如有更大显存可自行调大
- **patch size 与数据增强**：论文提到 512×512，但未说明是否第一阶段也使用随机裁剪和翻转，我们参照第二阶段的增强策略补充
- **验证集划分**：论文未提供 SICE 数据集的 train/val 划分方式，需要用户自行划分
- **验证脚本 `evaluate_stage1.py`**：完全由我们补充，论文和原始仓库均无相关内容
- **定量验证指标**（反射率一致性、光照图标准差）：论文未提出这些指标，由我们设计用于辅助判断分解质量
  > **Reflectance consistency** = `mean(|R1 - R2|)`：同场景不同曝光的反射率差异，越接近 0 说明分解越正确。
  > **Illumination std** = `std(L)`：光照图像素标准差，越小说明光照图越平滑、无内容残留。
  > 论文验证分解质量的方式是 Figure 3 中的目视对比，从未对 R、L 单独提出定量指标。因此这两个指标仅作为辅助参考，实际判断仍以目视检查为准。

## 概述

第一阶段使用 SICE 数据集中的多曝光配对图像，训练**编码器（Encoder）**、**内容转移分解网络（CTDN）** 和**解码器（Decoder）**。扩散模型（UNet）**不参与**本阶段。

训练目标是学习潜在空间的 Retinex 分解，使其能够生成：

- **内容丰富的反射率图（R）**：包含场景固有内容信息，在不同曝光下保持一致
- **无内容的光照图（L）**：仅表示光照条件，不包含任何内容信息

### 网络架构

```text
图像 ──► 编码器 (feature_pyramid + channel_down) ──► 潜在特征 (3通道, H/8 × W/8)
                                                          │
                                                          ▼
                                                   Retinex_decom (CTDN 核心)
                                                    ├─ 交叉注意力 (CrossAttention)
                                                    └─ 自注意力 (SelfAttention)
                                                          │
                                                    ┌─────┴─────┐
                                                    ▼           ▼
                                              反射率图 R      光照图 L
                                               (3通道)        (3通道)

潜在特征 ──► 解码器 (channel_up + 上采样 + 跳跃连接) ──► 重建图像
```

## 损失函数（论文 Section 3.4, Eq.7-9）

| 损失 | 公式 | 作用 |
|------|------|------|
| **L_con**（Eq.7） | `\|\| I - D(E(I)) \|\|_2` | 编码器-解码器重建保真度 |
| **L_rec**（Eq.8） | `Σ_i Σ_j \|\| F_j - R_i ⊙ L_j \|\|_1` | 分解后的 R、L 能重建原始特征（i,j∈1,2，共 4 项） |
| **L_ref**（Eq.9） | `\|\| R_1 - R_2 \|\|_1` | 同一场景不同曝光下反射率一致性 |
| **L_ill**（Eq.9） | `\|\| ∇L · exp(-λ_g · ∇R) \|\|_2` | 光照图局部平滑（边缘感知） |

**总损失**：`L = λ_con · L_con + L_dec`，其中 `L_dec = L_rec + λ_ref · L_ref + λ_ill · L_ill`

## 数据集准备

第一阶段使用 [SICE 数据集](https://github.com/csjcai/SICE)，该数据集提供同一场景在不同曝光条件下的多张图像。

### 目录结构

```text
/path/to/SICE_dataset/
├── sice_train.txt
├── sice_val.txt
└── images/
    ├── scene1/
    │   ├── 1.png    （欠曝光）
    │   ├── 2.png    （中等曝光）
    │   └── 3.png    （过曝光）
    ├── scene2/
    │   └── ...
    └── ...
```

### 文件列表格式

`sice_train.txt` / `sice_val.txt` 每行包含同一场景的**两张不同曝光图像**路径，以空格分隔：

```text
/path/to/images/scene1/1.png /path/to/images/scene1/3.png
/path/to/images/scene2/2.png /path/to/images/scene2/4.png
...
```

可以从每个场景的多曝光图像中任意组合配对。关键要求是**两张图像必须来自同一场景**——这是反射率一致性损失（L_ref）能够生效的前提。

## 配置参数

配置文件：`configs/stage1.yml`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.data_dir` | `/data/Image_restoration/SICE_dataset` | SICE 数据集根目录 |
| `data.patch_size` | 512 | 训练裁剪尺寸 |
| `training.batch_size` | 4 | 批量大小（论文为 12，因显存约束调为 4，详见上方说明） |
| `loss.lambda_con` | 0.01 | L_con 权重（论文 λ_1） |
| `training.n_iters` | 100000 | 总训练迭代次数 |
| `optim.lr` | 1e-4 | 初始学习率 |
| `training.lr_decay_factor` | 0.8 | 学习率衰减系数 |
| `training.lr_decay_every` | 20000 | 学习率衰减间隔（步数） |
| `loss.lambda_ref` | 0.1 | 反射率一致性损失权重 |
| `loss.lambda_ill` | 0.01 | 光照平滑损失权重 |
| `loss.lambda_g` | 10.0 | 光照平滑中的边缘感知强度 |

## 训练

```bash
# 从头开始训练
python train_stage1.py --config stage1.yml

# 从检查点恢复训练
python train_stage1.py --config stage1.yml --resume ckpt/stage1/model_step_50000.pth.tar
```

### 输出

检查点保存在 `ckpt/stage1/` 目录下：

```text
ckpt/stage1/
├── stage1_weight.pth.tar            # 最新权重（兼容第二阶段加载格式）
├── model_step_5000.pth.tar          # 定期保存的检查点（可恢复训练）
├── model_step_10000.pth.tar
├── ...
└── model_final.pth.tar              # 最终检查点（可恢复训练）
```

- `stage1_weight.pth.tar`：格式为 `{'model': state_dict}`，可直接被第二阶段的 `load_stage1()` 方法加载
- `model_step_*.pth.tar` / `model_final.pth.tar`：格式为 `{'step', 'state_dict', 'optimizer', 'scheduler'}`，用于恢复第一阶段训练

## 验证分解效果

训练完成后，使用 `evaluate_stage1.py` 可视化分解结果，验证模型是否正确学习了 Retinex 分解。

### 三种验证模式

```bash
# 模式 1：对单张图像进行分解，查看 R、L 和重建结果
python evaluate_stage1.py \
    --input_dir /path/to/test_images/ \
    --ckpt ckpt/stage1/stage1_weight.pth.tar

# 模式 2：对同一场景的两张不同曝光图像进行配对分解对比
python evaluate_stage1.py \
    --input_pair /path/to/scene1/dark.png /path/to/scene1/bright.png \
    --ckpt ckpt/stage1/stage1_weight.pth.tar

# 模式 3：批量配对验证（使用与训练相同格式的文件列表）
python evaluate_stage1.py \
    --pair_list /path/to/sice_val.txt \
    --ckpt ckpt/stage1/stage1_weight.pth.tar
```

### 输出内容

结果默认保存在 `results/stage1_eval/` 下：

**单张图像模式**（每张图生成一个子目录）：

| 文件 | 内容 |
|------|------|
| `input.png` | 原始输入图像 |
| `latent_feature.png` | 编码后的潜在特征（上采样可视化） |
| `reflectance_R.png` | 反射率图 R（应包含丰富的场景内容） |
| `illumination_L.png` | 光照图 L（应为平滑、无内容结构） |
| `reconstructed.png` | 重建图像 D(E(I))（应接近原始输入） |
| `comparison.png` | 以上五张图的横向对比拼接 |

**配对模式**（额外生成）：

| 文件 | 内容 |
|------|------|
| `reflectance_R1.png` / `R2.png` | 两张图的反射率（应高度相似） |
| `illumination_L1.png` / `L2.png` | 两张图的光照（应不同但均无内容） |
| `reflectance_diff_5x.png` | 反射率差异图 \|R1-R2\|（放大 5 倍，应接近全黑） |
| `comparison_pair.png` | 3×3 网格对比拼接 |

### 判断标准

分解效果合格的关键指标：

1. **反射率图（R）应包含丰富内容**：纹理、边缘、颜色等细节清晰可见
2. **光照图（L）应无内容结构**：仅展示亮度分布，看不到场景物体的轮廓
3. **配对反射率应高度一致**：同一场景不同曝光下，R1 和 R2 视觉上几乎一样
4. **重建图像接近原图**：D(E(I)) 与输入 I 无明显失真

脚本同时输出定量指标：

- **Reflectance consistency**（反射率一致性，越小越好）：`mean(|R1 - R2|)`
- **Illumination std**（光照图标准差，越小说明越平滑/无内容）：`std(L)`

### 对比参考

可参照论文 Figure 3 中的分解结果。合格的分解效果应类似于 Figure 3(b)（本方法的潜在空间分解），而非 Figure 3(a) 中其他方法在图像空间分解后光照图仍残留内容的情况。

## 衔接第二阶段

第一阶段训练完成后，`ckpt/stage1/stage1_weight.pth.tar` 会被第二阶段自动加载：

```python
# models/ddm.py 中 Net.__init__():
self.decom = self.load_stage1(CTDN(), 'ckpt/stage1')
```

然后运行第二阶段训练：

```bash
python train.py --config unsupervised.yml
```

第二阶段会**冻结** CTDN 的所有参数，仅训练扩散模型（UNet）。

## 文件说明

| 文件 | 作用 |
|------|------|
| `train_stage1.py` | 第一阶段训练入口脚本 |
| `evaluate_stage1.py` | 第一阶段分解效果验证与可视化脚本 |
| `configs/stage1.yml` | 第一阶段超参数配置 |
| `datasets/sice_dataset.py` | SICE 配对数据集加载器 |
| `models/decom.py` | CTDN 模型（编码器 + Retinex 分解 + 解码器） |
