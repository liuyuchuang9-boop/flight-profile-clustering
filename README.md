# flight-profile-clustering

Clustering algorithms for aero-engine flight mission profile and load spectrum analysis.

本仓库用于构建“航空发动机飞行任务剖面/载荷谱聚类分析”的可复现实验框架。当前建议采用“双基线”路线：

1. **物理/损伤特征聚类**：基于任务持续时间、阶段占比、转速/温度/高度/马赫数统计量、超限时长、雨流循环、Miner 损伤、等效载荷等任务级特征，复现 KMeans、GMM、HDBSCAN、OPTICS 等传统基线。
2. **多变量时间序列聚类**：基于等长或阶段对齐后的多变量序列，复现 DTW / Soft-DTW 聚类，并输出 barycenter 与最近真实任务 medoid。
3. **异常任务识别**：使用 HDBSCAN / OPTICS 在损伤加权特征空间或预计算 DTW 距离矩阵上识别异常飞行任务。

## 1. 安装环境

推荐使用 Python 3.10 或 3.11。

### 方案 A：pip

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell 可用 .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 方案 B：conda

```bash
conda env create -f environment.yml
conda activate flight-profile-clustering
```

## 2. 目录结构

```text
flight-profile-clustering/
├─ README.md
├─ requirements.txt
├─ requirements-optional.txt
├─ environment.yml
├─ .gitignore
├─ data/
│  └─ README.md
├─ docs/
│  └─ algorithm_selection_2026_06_01.md
└─ src/
   ├─ feature_baselines.py
   ├─ dtw_baselines.py
   ├─ anomaly_detection.py
   └─ utils.py
```

## 3. 推荐复现顺序

### 第一步：物理/损伤特征聚类

准备一个任务级特征表，例如：

```text
mission_id,duration,hover_ratio,climb_ratio,cruise_ratio,max_Ng,mean_Ng,max_T4,mean_T4,rainflow_count,miner_damage,equivalent_load
```

运行：

```bash
python src/feature_baselines.py --input data/features.csv --output outputs/feature_labels.csv --n-clusters 4
```

### 第二步：DTW / Soft-DTW 时间序列聚类

准备一个 NumPy 三维数组：

```text
shape = (n_missions, sequence_length, n_variables)
```

例如变量可以是：高度、马赫数、低压转速、高压转速、燃气温度、载荷参数、损伤代理量等。

运行：

```bash
python src/dtw_baselines.py --input data/series.npy --output-dir outputs/dtw --n-clusters 4 --metric dtw
python src/dtw_baselines.py --input data/series.npy --output-dir outputs/softdtw --n-clusters 4 --metric softdtw
```

### 第三步：异常任务识别

```bash
python src/anomaly_detection.py --input data/features.csv --output outputs/anomaly_labels.csv --method hdbscan
python src/anomaly_detection.py --input data/features.csv --output outputs/optics_labels.csv --method optics
```

## 4. 论文改进方向

最值得后续写成论文创新点的方向是：

> 损伤加权多变量 DTW / Soft-DTW + medoid / 真实代表任务约束 + 物理阶段约束。

该路线比单纯 KMeans 或单纯原始序列聚类更贴近航空发动机载荷谱编制与加速试车谱设计：既保留飞行任务剖面的时间形态，又引入转速、温度、雨流循环、Miner 损伤等寿命相关信息。

## 5. 注意事项

- 本仓库不直接复制第三方库源码，而是通过 `requirements.txt` 和 `environment.yml` 管理依赖。
- 第三方库源码应由 `pip` / `conda` 安装，避免许可证、版本维护和代码污染问题。
- 后续真实科研数据建议放在 `data/` 目录，但不要上传涉密、敏感或未经授权的数据。
