# 推理指南 —（NTIRE 2026 RAIM Track2）

**队伍：** whu-vip  
**联系方式：** whuyuanxy@gmail.com / yuanxiyuan@whu.edu.cn  
**模型架构：** 基于 AFUNet，引入空间特征变换（SFT）与自适应门控映射（Gated Map）模块，专为动态场景多曝光 HDR 融合设计。

---

## 1. 环境配置 (专为 V100 等测试机优化)

为了确保在主办方的测试环境（如 NVIDIA V100 GPU）中一次性顺利跑通，我们强烈建议使用 **CUDA 11.8** 版本的 PyTorch。

我们提供了一键配置脚本。请在 `WHU-VIP_FinalSubmission` 根目录下运行：
```bash
bash Scripts/setup_env.sh
```

**或者，您也可以手动执行以下命令：**

```bash
# 1. 创建并激活虚拟环境
conda create -n whu_vip python=3.10 -y
conda activate whu_vip

# 2. 安装兼容 V100 的 PyTorch
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. 安装核心依赖
pip install -r requirements.txt
```

---

## 2. 预训练权重

推理所需的最佳模型权重已包含在提交文件中，位于：
```text
PretrainedModels/WHU_VIP_AFUNet_SFT_GM.pth
```

---

## 3. 数据准备与输入格式

每个测试场景需放置在 `--input_root` 下的一个子文件夹中，以三位数字命名（如 `001`～`100`）。  
为了平衡计算效率与信息丰富度，本模型仅需输入 **3 张** 不同曝光的 JPEG 图像即可完成高质量融合：

```text
<input_root>/
├── 001/
│   ├── 0.jpg    # 欠曝光帧
│   ├── 2.jpg    # 正常曝光帧 (参考帧)
│   └── 4.jpg    # 过曝光帧
├── 002/
│   └── ...
```

---

## 4. 运行推理

我们推荐使用打包好的 Shell 脚本进行一键推理。您只需打开 `Scripts/test.sh`，修改顶部的 `INPUT_DIR` 和 `SAVE_DIR` 路径，然后运行：

```bash
bash Scripts/test.sh
```

### 完整参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_root` | *(必填)* | 测试场景根目录 |
| `--checkpoint` | *(必填)* | 模型权重路径 |
| `--save_root` | *(必填)* | 输出目录 |
| `--gpu_list` | `[0]` | 使用的 GPU 编号列表（支持多卡并发，如 `0 1 2 3`） |
| `--whole_image` | `True` | 整图推理开关（遇 OOM 会自动降级为分块推理） |
| `--zip_name` | `submission_whu_vip.zip` | 最终输出的提交压缩包名称 |

---

## 5. 推理策略与显存说明

本代码内置了极致的显存自适应安全机制：
- **默认（整图推理）：** 将输入 padding 至 128 的倍数后，进行单次无缝前向推理。对于 1528×2040 分辨率，单卡约需 **24 GB 显存**（如 RTX 3090 / 4090 / A100）。
- **自动回退（分块推理 Fallback）：** 若程序检测到当前显卡（如 16GB 的 V100）发生 `CUDA Out of Memory`，将**自动回退**为分块模式。图像会沿长边切分为 2 个带重叠的 patch（长边 1280），分别推理后使用线性渐变掩码加权融合。此模式下显存峰值仅需约 **15 GB**。

---

## 6. 输出说明

推理完成后，在 `--save_root` 目录下将生成：
1. **100 张重构的 HDR 图像**：命名为 `001.jpg`～`100.jpg`。
2. **readme.txt**：自动生成，包含测试设备的单张平均推理耗时和额外说明。
3. **提交压缩包 (submission_whu_vip.zip)**：自动将上述图像和 readme.txt 打包，可直接用于 Phase 3 提交。

---

## 7. 注意事项

- 模型推理默认启用 **AMP 自动混合精度 (FP16)**。
- 多卡并行推理使用了 Python `multiprocessing` 的 `spawn` 启动方式，每张 GPU 独立分配一个子进程，互不干扰，极大提升了测试吞吐量。
- 建议始终在 `WHU-VIP_FinalSubmission/` 根目录下执行所有命令，以确保模块的正常导入。
