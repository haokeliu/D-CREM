# CREM / D-CREM 固定约定（Protocol v2）

最后更新：2026-08-10

本文件归档 Protocol v2 中已经锁定、通常不随实验批次变化的数学、数据、环境和产物约定。公开发布版不包含内部实验待办与进度；结果由 `results/` JSON 保存并按需汇总。

除非方法定义或 Protocol 版本正式变更，否则不要重复或分叉这些内容；如需修改本文件，必须同步检查代码、测试、已有 JSON 和论文表述是否仍一致。

## 1. D-CREM 数学约定

### 1.1 P/W 符号

对标签 `k`：

- `W[:, k]` 是分类器向量；
- 代码存储的 `P[:, k]` 与分类器同向；
- 几何 reciprocal point 为 `r_k = -P_k`；
- classifier induction 在代码记号下是 `P_k ≈ W_k`，在几何记号下是 `r_k ≈ -W_k`；
- warm-up 后初始化 `P=W`，不是 `P=-W`。

因此：

```text
L_coupling = λ3/2 ||W-P||²_F
d_ik² = ||f_i-r_k||²
      = ||f_i+P_k||²
      = ||f_i||² + ||P_k||² + 2 f_iᵀP_k
```

最后一式使用真实特征范数，L2 norm 开启或关闭时都成立。

### 1.2 统一目标函数

对 `N` 个训练样本，标签 `k` 的正例数为 `n_k`：

```text
L = 1/(2N) ||FW + 1bᵀ - Y||²_F
  + λ1/2 ||W||²_F
  + λ2/2 tr(W L_C Wᵀ)
  + λ3/2 ||W-P||²_F
  + α Σ_k 1/n_k Σ_{i:y_ik=1} max(0, 1 + R_k² - ||f_i+P_k||²)
  + β L_unif(F)
  + γ L_div(P)
```

固定缩放规则：

- 分类损失按样本数平均；
- 开放空间损失按每类训练正例数归一化；
- mini-batch 使用训练折正例 prevalence；
- 显式 W/P/R/E 正则不再通过 AdamW weight decay 重复施加；
- Mode A 和 Mode B 优化同一个目标函数。

### 1.3 Mode B 的 W/b 闭式块

固定特征、P 和标签 Laplacian，令：

```text
F_c = F - mean(F)
Y_c = Y - mean(Y)
```

W 满足 primal Sylvester 方程：

```text
[F_cᵀF_c/N + (λ1+λ3)I] W + W(λ2 L_C)
    = F_cᵀY_c/N + λ3 P
```

求解后 `b = mean(Y-FW)`。Mode B 每隔 `T_sylvester` 个 epoch 精确更新 `W,b`；encoder、P、R 和标签嵌入 E 使用持久化 AdamW 更新。W/b 不进入梯度优化器。

### 1.4 P/R 梯度

```text
δ_ik = 1[1 + R_k² - ||f_i+P_k||² > 0]

∇_{P_k} L = λ3(P_k-W_k)
             - 2α/n_k Σ_i δ_ik(f_i+P_k)

∂L/∂R_k = 2αR_k/n_k Σ_i δ_ik
```

活动 hinge 会给 P 子问题引入负二次项，旧的反向 hinge 闭式 P 更新不能复用。Protocol v2 中 P/R 在两个 Mode 中都使用梯度更新。

### 1.5 CREM nominal/effective 参数

- 命令行保留 MATLAB 拼写 `lamda`；
- nominal 是论文和命令行参数，effective 是保留一次 MATLAB swap 后进入更新式的参数；
- swap 只允许在 `effective_from_nominal()` 中执行一次；
- 所有上游入口传 nominal 参数；
- JSON 同时记录 nominal/effective；
- `--legacy-dataset-params` 只用于兼容，不能生成 Protocol v2 正式结果。

### 1.6 论文精简核心的 classifier-induced 特例

论文正式 `paper_core` 不再把 `P` 作为自由参数，而是精确定义
`P := W`，几何 reciprocal point 为 `r_k=-W_k`。四轮 validation-only
开发实验均未找到跨数据集稳定的独立 P 监督，因此不得把自由 P、W-P coupling
或相关失败候选写成论文贡献。正式核心目标为：

```text
L_core = 1/(2N) ||FW + 1bᵀ - Y||²_F
       + λ1/2 ||W||²_F
       + β L_unif(F)
```

开放集主评分仍为 classifier-induced reciprocal distance：

```text
d_ik² = ||f_i + W_k||²
```

paper-core 固定 `λ2=λ3=α=γ=0`、不执行 warm-up；代码使用
`--classifier-induced-reciprocal` 保证评分直接读取 `W`，自由 reciprocal bank
不进入损失或优化器。Mode B 的闭式块相应退化为带 bias 的 ridge 解：

```text
[F_cᵀF_c/N + λ1 I] W = F_cᵀY_c/N,
b = mean(Y-FW).
```

Mode A 与 Mode B 优化相同的 `L_core`；区别仅是 W/b 使用 AdamW 联合更新还是
周期性闭式更新。S1 消融保持训练完全不变，只把 OSR 主评分从 reciprocal
distance 切换为 classifier logits。

## 2. 数据与评估协议

### 2.1 样本划分

```text
原始样本
  ├─ train       40%
  ├─ validation  10%
  └─ test        50%
```

同一 `(dataset, known_ratio, seed)` 下所有方法必须使用相同样本索引。`known_ratio` 只控制已知/未知标签比例，不改变样本比例。

### 2.2 防止数据泄漏

以下过程只能在训练折拟合，再应用于 validation/test：

- 特征筛选；
- 标准化均值和标准差；
- 模型参数；
- Firth 标定器；
- 训练超参数。

validation 负责模型选择和 top-K 选择。test 必须在所有选择锁定后评估一次，不得扫描 K、选择 checkpoint 或反向修改参数和候选网格。

### 2.3 数据与缓存边界

| 路径 | 处理规则 |
|---|---|
| `datasets_raw/*.arff`, `*.xml` | 原始数据，不修改 |
| `datasets/*.mat` | `main.py` 的旧复现输入，不用于最终 v2 主实验 |
| `cache/*_full.mat` | Protocol v1 缓存，不读取 |
| `cache/*_protocol_v2.mat` | 表格数据有效缓存，可自动重建 |
| `cache/voc2007_*.mat` | 冻结图像特征，缺失时才重新提取 |
| `results/` | JSON 入仓；大体积分析缓存由 Git 忽略 |

### 2.4 结果目录与 JSON 最低字段

```text
results/
├── crem_v2/<dataset>/known_ratio=<ratio>/seed<seed>.json
├── dcrem/<dataset>/protocol_v2_modeA_r<ratio>/seed<seed>.json
├── dcrem/<dataset>/protocol_v2_modeB_r<ratio>/seed<seed>.json
├── dcrem/<dataset>/protocol_v2_paper_core_mode<mode>_r<ratio>/seed<seed>.json
├── dcrem/<dataset>/protocol_v2_ablation_modeA_core_<ID>_r<ratio>/seed<seed>.json
├── dcrem/<dataset>/protocol_v2_ablation_modeB_core_<ID>_r<ratio>/seed<seed>.json
├── dcrem/<dataset>/protocol_v2_sensitivity_modeB_core_<ID>_r<ratio>/seed<seed>.json
├── dcrem/voc2007/protocol_v2_paper_core_image_<arm>_resnet50_modeB_r<ratio>/seed<seed>.json
├── tables/ablation_modeA_core_effects.json
├── tables/ablation_modeB_core_effects.json
├── tables/modeB_core_sensitivity.json
├── baselines_v2/{ocsvm,iforest,slan}/<dataset>/known_ratio=<ratio>/seed<seed>.json
└── analysis_cache_protocol_v2/
```

没有 `protocol_v2` 标记的 D-CREM 结果拒绝汇总。每个正式 JSON 至少记录：

- Protocol 版本和 0.4/0.1/0.5 三折比例；
- `preprocessing_fit: train_only`；
- validation 选择的 K 和搜索明细；
- seed、确定性环境和代码 revision；
- 配置、指标、运行时间；
- CREM 结果的 nominal/effective 参数。

## 3. 环境与工程边界

- 正式环境：conda `pytorch`、Python 3.9、NumPy 1.26.x；
- 当前参考环境：torch 2.8.0 + CUDA 12.8；
- `crem/` 不新增 NumPy/SciPy/scikit-learn 之外的依赖；
- 特征按 `N×d`；对外 target 按 `Q×N`、取值 `±1`；
- 不用 `main.py` 生成论文结果；
- 不修改原始数据来改善指标；
- 不恢复或汇总 Protocol v1 结果；
- 删除失败 run 必须精确到单个 JSON，不能清空有效结果根目录。

## 4. 论文模型与消融固定规则

- 论文正式 D-CREM 使用 1.6 节的 classifier-induced 精简核心：分类损失、
  `λ1` 正则和 feature uniformity；固定 `P:=W`、`λ2=λ3=α=γ_div=0`，且不执行
  warm-up；
- correlation、open-space、diversity 和 warm-up 的实现及历史结果仅用于复现
  已废弃诊断，不再属于论文方法、正式主实验或消融；
- 完整模型先用 train/validation 锁定配置；
- 每个变体只改变一个预先声明的模块；
- AUROC 是唯一 OSR 主指标，AUPR 和 macroAUC 为补充指标；
- 正式 Mode A 解释性消融和 Mode B 确认性消融都只包含 full、N1、E1、S1、U1；
- 旧 Mode B、slashdot、`r=0.5` 单点消融只保留作诊断，不进入新的 Mode B
  确认性消融统计；
- 已完成的旧十变体 Mode A 消融也转为废弃诊断，不进入改版论文统计；
- 同一设置的 full 与变体必须使用成对 seed、相同划分和相同训练预算。
- 敏感性实验是预注册的单因素稳健性分析；每个配置只改变一个参数，不得根据
  test 结果挑选新的默认值或替换正式主结果。
- Protocol v2 主矩阵继续锁定 `T_block=10`、`λ1=1`、`β=0.1` 和 `d'=128`。
  Mode B 的 U1 消融与 `β` 敏感性结果只能用于报告优化器—组件交互，不能据此
  回溯修改默认值；若后续采用 `β=0` 或其他新配置，必须另立协议并用独立证据
  重新验证。
- VOC2007 端到端扩展只读取原图，使用 ImageNet 预训练 ResNet-50 与 128 维
  可学习投影；训练增强只作用于 train，validation/test 使用固定变换。同一
  `(known_ratio, seed)` 的 end-to-end 与 frozen-backbone 对照必须复用完全相同的
  样本索引和标签划分，K 仍只由 validation 选择。

## 5. 已知边界

- 原论文的 `rcvsubset2-2` 缺失，当前使用 6 个表格数据集；
- bibtex 核 Sylvester 较慢；
- enron 等数据集的标签名为代码，semantic C 只适合 bibtex；
- SLAN 是官方 MATLAB 算法的 Python 移植，论文需注明数值求解器差异；
- MUENL-F 已移植并决定纳入正式主矩阵；运行成本只影响调度，不再作为省略基线的理由；
- 正式实验必须在 `pytorch` conda 环境执行。
