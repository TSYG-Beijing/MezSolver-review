# Toyota Yaris 116 整车碰撞三阶段训练示例

本目录用于在已集成审阅版 MezSolver 的 PhysicsNeMo 中，从头训练 Toyota R02 的 U1、U2、U3 位移模型。它提供实际训练分区、监督文件、固定验证面板和启动器。116 个工况按 92 / 12 / 12 划分为训练、验证、测试，每个工况 31 帧，碰撞速度为 40–60 km/h。模型使用 1,488,757 个节点、5,617 个节点训练区域，每个车轮有 6 个同心环。该目录供内部实验使用，数据与几何文件的对外发布由权利人决定。

## 从桌面包到训练

以下命令在 Linux、Bash 和带 CUDA 的独立 Python 环境运行。先将整个桌面目录复制到训练机，进入该目录。四卡是本实验的原始配置。当前实例上的空闲卡需自行选择。

### 1　准备审阅版源码

```bash
export BUNDLE="$(pwd -P)"
export PHYSICSNEMO_ROOT="$HOME/physicsnemo-mezsolver-r02"
export RUN="$HOME/runs/toyota-r02-integrated"
git clone https://github.com/NVIDIA/physicsnemo.git "$PHYSICSNEMO_ROOT"
git -C "$PHYSICSNEMO_ROOT" switch --detach f32f11b3c63b683af3b2af1afce49a105019974c
python "$BUNDLE/source/apply_review.py" "$PHYSICSNEMO_ROOT"
export EXAMPLE="$PHYSICSNEMO_ROOT/examples/structural_mechanics/partitioned_mesh_displacement"
python -m pip install -e "$PHYSICSNEMO_ROOT"
python -m pip install -r "$EXAMPLE/requirements.txt"
python -c "import torch; from physicsnemo.experimental.models.partitioned_mesh import TemporalPredictor; print(torch.__version__, torch.cuda.is_available())"
```

首次安装请选独立的新源码目录。已有固定版本检出可直接指定 `PHYSICSNEMO_ROOT` 并从 `apply_review.py` 开始。它会检查随包 ZIP、补丁和检出目录的 46 个源文件。固定基线的 Python 范围是 3.11 至 3.14；请使用与该基线依赖和驱动匹配的 CUDA 版 PyTorch。既有云实例系统环境和历史训练环境可能导入另一版 PhysicsNeMo。默认时序扫描为 `parallel`，安装步骤无需 Mamba 扩展。

### 2　连接真实响应数据并预检

```bash
# 本次在现有实例上实际读过的镜像。新实例请换成自己的完整数据目录。
export DATA=/dev/shm/yaris_direct_io_20261123
test -d "$DATA/cases" && ln -s "$DATA" "$BUNDLE/training_hdf5"
python "$BUNDLE/check_inputs.py"
```

建链接只需执行一次。预检成功时会显示 116 个工况的 HDF5 结构、掩码与单元活动检查通过，并实际抽读 F001 第 1 帧的两个位移块（帧号从 0 开始）。仅核对桌面文件可执行 `python check_inputs.py --assets-only`。预检不会读取测试位移数值。默认清单使用固定的相对目录 `training_hdf5`，整包搬移可保留同一份清单签名。

所需外部数据为 `cases/<case>.h5` 共 116 份、`training_masks.h5`、`element_activity.h5`、`Toyota_yaris_116_front_v4.h5` 和 `dataset_manifest.json`。仅 116 份工况响应就约 **101.70 GiB**。后两份小型身份核验文件也随附在 `provenance/dataset_identity/`，不能替代工况或活动掩码。

**当前实例的读盘情况。** 原始 NAS 上能列出文件和 HDF5 结构，从 F001 的第一块位移读取会报 `wrong B-tree signature`。同一工况从上述镜像读取成功，F001 文件 SHA-256 与原数据校验表相同。完整核验范围见 [真实数据读取记录](verification/actual_data_preflight.json)。`/dev/shm` 在重启后可能清空；使用新机器时先找已校验的持久数据副本。

如需重建镜像，下列是**单独的可选数据准备步骤**。仅在 Linux 上运行，准备一个与源目录分开的新目标目录，预留超过完整数据集的可用空间；程序会复制全部工况和辅助文件。源路径、校验表 `SHA256SUMS` 和底层存储须支持 GNU `dd iflag=direct`。工况摘要不符时，程序报告失败。成功后将 `DATA` 指向镜像并重新执行预检。

```bash
export ORIGINAL=/root/nas/usb_jida_20261109/A100_ZHANGYUAN/ZhangYuan/Toyota_yaris_116/training_hdf5
export NEW_MIRROR=/path/on/local-storage/toyota116-verified
python "$BUNDLE/data_tools/stage_data.py" "$ORIGINAL" "$NEW_MIRROR" "$BUNDLE/mirror-progress.json" --workers 4
```

### 3　打印命令并运行

```bash
# 按本机可用卡编号设置。例如使用四张 0–3 号卡。
export CUDA_VISIBLE_DEVICES=0,1,2,3
python "$BUNDLE/launch/run.py" --stage all --gpus 4 --output "$RUN" --dry-run
python "$BUNDLE/launch/run.py" --stage all --gpus 4 --output "$RUN"
```

`--dry-run` 只打印三个实际的 torchrun 命令。正式运行后，Stage 1 随机初始化，Stage 2 读取 `$RUN/stage1/best.pt`，Stage 3 读取 `$RUN/stage2/best.pt`。每阶段有单独目录，只有完成标志与验证最优权重相符才会进入下一阶段。指定 `--stage 1`、`--stage 2` 或 `--stage 3` 可以单独启动。

| 阶段 | 训练对象 | 计划轮数 | R02 参数 |
| --- | --- | ---: | --- |
| Stage 1 | 空间自编码器、真实相邻两帧的一步预测器 | 500 | 每轮 256 步，每卡 16 个采样区；车轮监督预设、节点数和材料加权、区域注意力 4 头 |
| Stage 2 | 冻结空间编码器和解码器，训练初始场到整段轨迹的时序模型 | 900 | 每轮 256 步，每卡 2 个采样区、每步监督 4 个未来帧；2 层区域交换 + 2 层 SSM，中间再做一次同一时刻的邻域交换；接缝梯度权重 0.005 |
| Stage 3 | 联合更新时序模型和空间解码器，空间编码器冻结 | 200 | 每轮 256 步，每卡 2 个采样区；重构权重 0.1，接缝梯度权重 0.01 |

两个时序阶段按原计划使用等权 U1/U2/U3、0.4 的车轮批次比例、0.3 的接缝批次比例、车轮内部与接口边权重 0.2/0.1。独立的未除边长位移差损失权重为 0，接缝梯度项使用参考边长。训练验证使用固定的 215 个分区、59,296 个一般验证节点；车轮验证使用其对应的额外区域。500/900/200 是复现设置，不代表本包已经完成这三段训练。

历史设置的响应缓存上限为 Stage 1 每进程训练数据集实例 65,536 MiB，Stage 2/3 为 16,384 MiB；验证缓存设置为 8,192 MiB。多卡会增加主机内存需求。先试跑或在主机内存较少的实例上可覆盖缓存、并减少同时打开的工况。以下短跑使用独立目录：

```bash
python "$BUNDLE/launch/run.py" --stage all --gpus 4 --output "${RUN}-smoke" \
  --epochs 1 1 1 --steps-per-epoch 1 --response-cache-mb 0 \
  --validation-cache-mb 0 --open-cases 3
```

同一版本的训练中断后，保留原输出、GPU 进程数和启动参数，追加 `--resume`。例如默认充分训练的恢复命令如下。已完成的阶段会跳过，未完成的阶段读取自己的 `last.pt`。旧实验的断点使用旧清单和源码签名，本目录用于新的集成版训练。

```bash
python "$BUNDLE/launch/run.py" --stage all --gpus 4 --output "$RUN" --resume
```

## 训练后的验证和测试

```bash
python "$EXAMPLE/evaluate_components.py" --manifest "$BUNDLE/manifest.json" \
  --checkpoint "$RUN/stage1/best.pt" --split val --frames 8 15 22 30 --device cuda \
  --output "$RUN/stage1/val_components.json"
python "$EXAMPLE/evaluate_components.py" --manifest "$BUNDLE/manifest.json" \
  --checkpoint "$RUN/stage3/best.pt" --split test --full-mesh --device cuda \
  --output "$RUN/stage3/test_full_mesh.json"
python "$EXAMPLE/export_models.py" --checkpoint "$RUN/stage3/best.pt" --output "$RUN/exported_stage3"
```

分量指标为 U1/U2/U3 的 RMSE（mm）与相对 L2。Stage 1 计算重构和一步预测；Stage 2/3 的未来预测仅以初始场为观测。Stage 3 示例对全车、独立测试集评估，耗时会高于面板验证。

## 文件导航

| 路径 | 内容 |
| --- | --- |
| [manifest.json](manifest.json) | 116 个工况的 ID、切分、速度、时间网格、归一化及资产路径 |
| [assets/mesh_5617.npz](assets/mesh_5617.npz) | 训练实际读取的整车网格，含节点分区归属、核心索引、图邻接和参考特征 |
| [assets/region_sdf_5617.pt](assets/region_sdf_5617.pt) | 与上述节点和分区顺序对应的 SDF 特征 |
| [assets/training_region_ids.npy](assets/training_region_ids.npy) | 训练时紧凑区号对应的原几何区号，节点归属在网格文件里 |
| [assets/material_weights.json](assets/material_weights.json) | 与这份网格绑定的材料变形权重 |
| [assets/wheel_training_groups.npz](assets/wheel_training_groups.npz) 与 [assets/seam_pairs.npz](assets/seam_pairs.npz) | 车轮内部和接口监督；44,598 条跨分区壳接缝边 |
| [assets/validation_patches.json](assets/validation_patches.json) 与 [assets/validation_node_rows.npy](assets/validation_node_rows.npy) | 固定验证分区和节点行号 |
| [geometry/](geometry/) | ParaView 可打开的整车 VTU、节点/单元分区映射、部件/材料索引、轮轴/轮心/圆环定义及分区图。5,518 个几何单元区、5,542 个严格单元子区、5,617 个节点训练区是三种统计口径 |
| [source/MezSolver_public.zip](source/MezSolver_public.zip) | 完整审阅源码、基线补丁、许可证、API 与示例文档 |
| [launch/run.py](launch/run.py) / [check_inputs.py](check_inputs.py) | 三阶段启动与恢复；本地资产和真实工况预检 |
| [provenance/asset_conversion.json](provenance/asset_conversion.json) | 归档格式与公开格式的摘要映射；原始清单、材料、车轮与接缝元数据存于 `provenance/original/` |

网格、SDF、分区映射和固定验证节点使用原文件字节。为使公开版加载器直接读入数据，清单改为相对路径，材料、车轮与接缝文件改用公开格式标签并更新相应哈希；分组、权重、数组值、工况划分及归一化数值已核对。训练文件请使用根目录 `manifest.json` 和 `assets/`。

