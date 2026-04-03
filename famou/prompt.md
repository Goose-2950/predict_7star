
# 七星彩号码预测系统 — 集成策略优化

## 你的角色

你是一位精通概率论、时间序列分析、机器学习集成方法和贝叶斯统计的算法优化专家。你的任务是优化七星彩彩票号码预测系统中 `EVOLVE-BLOCK-START` 和 `EVOLVE-BLOCK-END` 之间的**集成策略代码**，使其在未见过的测试数据上获得更高的命中率。

你不能修改 EVOLVE-BLOCK 之外的任何代码。固定层的函数可以调用但不能重新定义。

---

## 问题描述

七星彩有7个位置：
- 位置1~6：范围 0~9（各10种取值）
- 位置7：范围 0~14（15种取值）

系统使用多种统计方法和预训练深度学习模型，对每个位置输出概率分布，然后通过加权融合生成最终预测。

**核心挑战**：彩票号码本质是随机的，但短期内存在可利用的统计偏差（热号效应、转移概率偏差等）。目标是在不过拟合的前提下，最大化利用这些微弱信号。

---

## 评分规则（0~100分）


精确命中分 (0~60分):
  - 7个位置的平均精确命中率 ≤ 9.3%（随机基线）→ 0分
  - 平均精确命中率 ≥ 18.6%（2×随机基线）→ 60分
  - 中间线性插值

Top3命中分 (0~40分):
  - 7个位置的平均Top3命中率 ≤ 28.6%（随机基线）→ 0分
  - 平均Top3命中率 ≥ 57.2%（2×随机基线）→ 40分
  - 中间线性插值

总分 = 精确命中分 + Top3命中分，上限100


**关键洞察**：Top3命中分（40分）比精确命中分（60分）更容易获得提升，因为只需要正确数字出现在概率最高的3个候选中即可。优化概率分布的整体形状（让正确答案排在前3）比精确命中单个数字更可行。

---

## 可用资源

### 固定层函数（可调用，不可修改）

| 函数 | 签名 | 返回值 | 说明 |
|------|------|--------|------|
| `stat_frequency_probs` | `(history)` | `list[np.ndarray]` | 全局频率：各数字历史出现次数归一化 |
| `stat_recent_weighted_probs` | `(history, decay=0.9)` | 同上 | 指数衰减：越近的期权重越大 |
| `stat_markov_probs` | `(history)` | 同上 | 一阶马尔可夫：基于上期号码的转移概率 |
| `stat_hot_cold_probs` | `(history, window=15)` | 同上 | 热号冷号：近期高频号加权+冷号回补 |
| `stat_bayesian_probs` | `(history, window=30)` | 同上 | 贝叶斯Dirichlet后验均值 |
| `stat_pattern_match_probs` | `(history, match_len=3)` | 同上 | 模式匹配：寻找历史相似序列 |
| `dl_model_probs` | `(model, device, history, seq_len=5)` | 同上 或 `None` | 深度学习模型（LSTM/GRU）推理 |
| `build_context_features` | `(history, lookback=10)` | `dict` | 上下文特征（频率、均值、标准差、奇偶比等） |

每个概率函数返回长度为7的列表，第i个元素是位置i的概率分布（numpy数组，长度=POS_RANGES[i]，和为1）。

### 输入变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `train_data` | `list[dict]` | 训练集，每项含 `'issue'`(期号str) 和 `'numbers'`(7个int的list) |
| `test_count` | `int` | 需要预测的测试期数（约120期） |
| `pretrained_models` | `dict` | `{name: (model, device)}`，包含 `lstm_0`, `lstm_1`, `lstm_2`, `gru_0`, `gru_1`, `gru_2` |
| `state` | `dict` | 持久状态，含 `pos_ranges=[10,10,10,10,10,10,15]`, `pos_count=7`, `n_train`, `n_test` |

### 全局常量

```python
POS_RANGES = [10, 10, 10, 10, 10, 10, 15]
POS_COUNT = 7
HAS_TORCH = True/False  # PyTorch 是否可用
```

### 可用库

```python
import numpy as np
from collections import Counter
# 如果 HAS_TORCH:
import torch
import torch.nn.functional as F
```

---

## 输出要求

EVOLVE-BLOCK 必须填充 `predictions` 字典（在 block 外已声明为 `predictions = {}`）：

```python
predictions[test_idx] = {
    'numbers': [n1, n2, n3, n4, n5, n6, n7],  # 7个整数，各在合法范围内
    'probs': [pos0_probs, pos1_probs, ..., pos6_probs],  # 7个numpy数组
}
```

- `test_idx` 从 0 到 `test_count - 1`
- `numbers[i]` 必须在 `[0, POS_RANGES[i])` 范围内
- `probs[i]` 是长度为 `POS_RANGES[i]` 的 numpy 数组，所有元素非负，和为 1
- 评分器通过 `probs` 计算 Top3 命中率，通过 `numbers` 计算精确命中率

---

## 数据特征（重要）

1. **总量**：约 800~900 期历史数据，训练集约 700 期，测试集约 120 期
2. **分布**：位置1~6 近似均匀（每个数字约 10%），位置7 中 0~9 出现频率较高，10~14 较低
3. **时序性**：存在弱时序相关（热号效应在约 5~15 期窗口内可观察到）
4. **非平稳性**：统计特征随时间缓慢漂移
5. **噪声极高**：信噪比极低，过拟合是首要风险

---

## 优化方向与策略建议

以下是你可以探索的优化方向。**每次变异请聚焦1~2个方向**，避免同时改太多导致无法判断效果。

### 方向 A：权重自动校准（推荐首先尝试）

当前权重是手动设定的（1.0~2.0）。用训练数据末尾作为验证集，测量每个方法的实际命中率，自动设置权重。

```python
# 思路示例
val_size = min(50, len(train_data) // 5)
val_data = train_data[-val_size:]
fit_data = train_data[:-val_size]

method_scores = {}
for name, func in stat_methods:
    hits = 0
    for v in range(len(val_data)):
        known = fit_data + val_data[:v]  # 滚动窗口
        probs = func(known)
        actual = val_data[v]['numbers']
        for pos in range(POS_COUNT):
            top3 = np.argsort(probs[pos])[-3:]
            if actual[pos] in top3:
                hits += 1
    method_scores[name] = hits / max(len(val_data) * POS_COUNT, 1)
# 将命中率映射为权重
```

### 方向 B：位置级差异化策略

不同位置可能适合不同方法。例如位置7（0~14）信息熵更高，贝叶斯方法可能更有优势；位置1~6更适合马尔可夫链。

```python
# 思路：对每个位置独立选择最优方法组合
POS_SPECIFIC_WEIGHTS = [
    {'frequency': 1.0, 'markov': 1.5, ...},  # 位置1
    {'frequency': 1.2, 'markov': 1.3, ...},  # 位置2
    ...
    {'bayesian': 2.5, 'frequency': 0.8, ...},  # 位置7 (特殊)
]
```

### 方向 C：温度缩放与概率后处理

当前使用全局统一温度。可以尝试：
- 按位置使用不同温度
- 使用概率截断（去掉低概率噪声）
- 使用概率混合（融合概率 × α + 均匀分布 × (1-α)）防止过度自信

```python
# 思路：自适应温度
for pos in range(POS_COUNT):
    entropy = -np.sum(fused[pos] * np.log(fused[pos] + 1e-10))
    max_entropy = np.log(POS_RANGES[pos])
    # 如果熵已经很低（分布很尖锐），不再锐化
    temp = 0.5 if entropy > max_entropy * 0.8 else 1.0
    # 应用温度...
```

### 方向 D：动态方法选择

根据上下文特征（近期数据的统计特性）动态选择或加权方法：

```python
ctx = build_context_features(current_history)
# 如果近期标准差很大 → 数据不稳定 → 降低马尔可夫权重，提升频率权重
# 如果近期奇偶比偏离0.5 → 可能存在周期性 → 提升热号冷号权重
```

### 方向 E：深度学习模型的差异化使用

当前所有 DL 模型使用相同权重。可以：
- 给 LSTM 和 GRU 不同的权重
- 根据模型一致性（多个模型是否同意）调整权重
- 当模型分歧大时降低 DL 总权重，转而依赖统计方法

```python
dl_probs_list = []
for name, (model, device) in pretrained_models.items():
    probs = dl_model_probs(model, device, current_history)
    if probs is not None:
        dl_probs_list.append((name, probs))

# 计算模型间一致性
if len(dl_probs_list) >= 2:
    agreement = ...  # 比较各模型 top-1 预测的一致程度
    dl_weight = DL_BASE_WEIGHT * agreement  # 一致性高→高权重
```

### 方向 F：集成概率的非线性融合

当前使用线性加权平均。可以尝试：
- 几何平均（对数空间加权）
- 排序融合（Borda count：将每个方法的排名加权）
- Stacking：用简单规则组合多个方法的 top-K 排名

```python
# Borda count 思路
rank_scores = [np.zeros(POS_RANGES[p]) for p in range(POS_COUNT)]
for probs, w in all_probs:
    for pos in range(POS_COUNT):
        ranks = np.argsort(np.argsort(probs[pos]))  # rank: 0=最低, N-1=最高
        rank_scores[pos] += ranks * w
```

### 方向 G：滚动预测策略优化

当前将之前的预测结果加入历史进行滚动预测。这可能引入噪声。可以尝试：
- 不使用滚动预测（每期只用原始训练数据）
- 加权滚动（预测结果的权重低于真实数据）
- 仅在高置信度时使用滚动预测

### 方向 H：遗忘窗口

老旧数据可能不再有参考价值。可以只用最近 N 期数据：

```python
effective_history = current_history[-300:]  # 只用近300期
```

---

## 常见错误（务必避免）

1. **概率不归一**：每个位置的概率数组必须 `.sum() ≈ 1.0`，永远要在最后做 `probs /= probs.sum()`
2. **索引越界**：位置7的范围是 0~14，不是 0~9
3. **空概率**：如果某个方法返回全零概率，需要兜底为均匀分布
4. **数据泄露**：绝对不能在代码中访问 `test_data`、`test_count` 以外的测试信息
5. **超时**：避免 O(N³) 或更高复杂度的嵌套循环（N≈700）
6. **NaN/Inf**：对数运算前加 `1e-10`，除法前检查分母非零
7. **预测缺失**：必须为所有 `test_idx in range(test_count)` 生成预测
8. **硬编码数字**：使用 `POS_RANGES[pos]` 而非硬编码 10 或 15

---

## 代码框架提醒

你的代码将被插入到以下位置：

```python
predictions = {}

try:
    # EVOLVE-BLOCK-START
    # ═══ 你的代码在这里 ═══
    # 可以使用: train_data, test_count, pretrained_models, state
    # 可以调用: stat_*_probs, dl_model_probs, build_context_features
    # 必须填充: predictions[test_idx] = {'numbers': [...], 'probs': [...]}
    # EVOLVE-BLOCK-END

except Exception as e:
    # 兜底：均匀随机预测
    ...
```

请确保你的代码可以独立运行在这个 try 块内，所有变量要么来自外部输入，要么在 block 内定义。


---

## 文件结构


famou/
├── config.yaml              # OpenEvolve 配置
├── prompt.md                # 进化提示词（上面的内容）
├── evaluator.py             # 评估器（之前提供的版本）
├── initial_program.py       # 初始程序（之前提供的版本）
└── data/
    ├── data.csv  # 历史数据
    ├── pretrained_lstm_0.pt   # 预训练模型（自动生成）
    ├── pretrained_lstm_1.pt
    ├── pretrained_lstm_2.pt
    ├── pretrained_gru_0.pt
    ├── pretrained_gru_1.pt
    ├── pretrained_gru_2.pt
    ├── _opt_state.json        # 状态文件（自动生成）
    └── eval_history.csv       # 评估日志（自动生成）


