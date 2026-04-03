"""
七星彩预测 — OpenEvolve 可进化版

结构:
  - 固定层: 数据接口、归一化、统计工具（不可修改）
  - EVOLVE-BLOCK: 集成策略（OpenEvolve 自动进化）
    ├─ 统计方法权重
    ├─ 模型组合逻辑
    ├─ 特征工程
    └─ 后处理策略

OpenEvolve 只修改 EVOLVE-BLOCK 内的代码。
"""

import numpy as np
from collections import Counter

# ═══════════════════════════════════════════════════════════════
#  固定层：常量 & 工具
# ═══════════════════════════════════════════════════════════════

POS_RANGES = [10, 10, 10, 10, 10, 10, 15]
POS_COUNT = 7

HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════
#  固定层：基础统计方法（返回概率分布）
# ═══════════════════════════════════════════════════════════════

def stat_frequency_probs(history):
    """频率法：各位置各数字的出现频率"""
    probs = []
    for pos in range(POS_COUNT):
        nv = POS_RANGES[pos]
        c = Counter(r['numbers'][pos] for r in history)
        w = np.array([c.get(v, 0) + 0.1 for v in range(nv)])
        probs.append(w / w.sum())
    return probs


def stat_recent_weighted_probs(history, decay=0.9):
    """指数衰减加权：近期数据权重更高"""
    n = len(history)
    probs = []
    for pos in range(POS_COUNT):
        nv = POS_RANGES[pos]
        sc = np.zeros(nv)
        for j in range(n):
            sc[history[j]['numbers'][pos]] += decay ** (n - 1 - j)
        sc += 1e-10
        probs.append(sc / sc.sum())
    return probs


def stat_markov_probs(history):
    """一阶马尔可夫链：转移概率"""
    if len(history) < 2:
        return stat_frequency_probs(history)
    last = history[-1]['numbers']
    probs = []
    for pos in range(POS_COUNT):
        nv = POS_RANGES[pos]
        matrix = np.ones((nv, nv)) * 0.01
        for j in range(1, len(history)):
            prev = history[j-1]['numbers'][pos]
            curr = history[j]['numbers'][pos]
            matrix[prev][curr] += 1
        for r in range(nv):
            matrix[r] /= matrix[r].sum()
        probs.append(matrix[last[pos]])
    return probs


def stat_hot_cold_probs(history, window=15):
    """热号冷号：近期高频号码加权"""
    recent = history[-window:]
    probs = []
    for pos in range(POS_COUNT):
        nv = POS_RANGES[pos]
        c = Counter(r['numbers'][pos] for r in recent)
        w = np.zeros(nv)
        for v in range(nv):
            freq = c.get(v, 0)
            if freq >= 2:
                w[v] = freq * 2.0
            elif freq == 0:
                w[v] = 1.5  # 冷号回补
            else:
                w[v] = 1.0
        w += 0.01
        probs.append(w / w.sum())
    return probs


def stat_bayesian_probs(history, window=30):
    """贝叶斯 Dirichlet 后验"""
    probs = []
    for pos in range(POS_COUNT):
        nv = POS_RANGES[pos]
        alpha = np.ones(nv)
        for rec in history:
            alpha[rec['numbers'][pos]] += 0.3
        for j, rec in enumerate(history[-window:]):
            alpha[rec['numbers'][pos]] += (1 + j / max(window, 1) * 2) * 0.7
        probs.append(alpha / alpha.sum())
    return probs


def stat_pattern_match_probs(history, match_len=3):
    """模式匹配：寻找历史相似序列"""
    if len(history) < match_len + 1:
        return stat_frequency_probs(history)
    query = [r['numbers'] for r in history[-match_len:]]
    sims = []
    for i in range(len(history) - match_len - 1):
        seg = [history[i+j]['numbers'] for j in range(match_len)]
        sc = sum(1 if seg[t][p] == query[t][p] else 0.5 if abs(seg[t][p]-query[t][p]) <= 1 else 0
                 for t in range(match_len) for p in range(POS_COUNT))
        sims.append((sc, i + match_len))
    sims.sort(key=lambda x: x[0], reverse=True)

    probs = [np.zeros(POS_RANGES[p]) for p in range(POS_COUNT)]
    for sc, ni in sims[:10]:
        if ni < len(history):
            for pos in range(POS_COUNT):
                probs[pos][history[ni]['numbers'][pos]] += sc + 0.01
    for pos in range(POS_COUNT):
        probs[pos] += 0.01
        probs[pos] /= probs[pos].sum()
    return probs


# ═══════════════════════════════════════════════════════════════
#  固定层：DL 模型推理（使用预训练模型）
# ═══════════════════════════════════════════════════════════════

def dl_model_probs(model, device, history, seq_len=5):
    """用预训练的 LSTM/GRU 模型预测概率分布"""
    if not HAS_TORCH or model is None:
        return None
    numbers = np.array([r['numbers'] for r in history])
    if len(numbers) < seq_len:
        return None
    seq = numbers[-seq_len:]
    x = torch.FloatTensor(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = [torch.softmax(out[p], -1).cpu().numpy()[0] for p in range(POS_COUNT)]
    return probs


# ═══════════════════════════════════════════════════════════════
#  固定层：特征构建工具
# ═══════════════════════════════════════════════════════════════

def build_context_features(history, lookback=10):
    """
    构建上下文特征向量（供 EVOLVE-BLOCK 使用）
    返回 dict 包含各种统计特征
    """
    recent = history[-lookback:]
    features = {}

    # 各位置最近出现频率
    for pos in range(POS_COUNT):
        nv = POS_RANGES[pos]
        nums = [r['numbers'][pos] for r in recent]
        c = Counter(nums)
        features[f'pos{pos}_freq'] = [c.get(v, 0) / len(nums) for v in range(nv)]
        features[f'pos{pos}_last'] = recent[-1]['numbers'][pos]
        features[f'pos{pos}_mean'] = np.mean(nums)
        features[f'pos{pos}_std'] = np.std(nums) if len(nums) > 1 else 0

    # 和值
    sums = [sum(r['numbers']) for r in recent]
    features['sum_mean'] = np.mean(sums)
    features['sum_std'] = np.std(sums)

    # 奇偶比
    last_nums = recent[-1]['numbers']
    features['odd_ratio'] = sum(1 for x in last_nums if x % 2 == 1) / POS_COUNT

    # 数据量
    features['n_history'] = len(history)
    features['n_recent'] = len(recent)

    return features


# ═══════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════

def generate_x(train_data, test_count, pretrained_models, state):
    """
    为测试集每一期生成预测。

    参数:
        train_data:         训练数据列表 [{issue, numbers}, ...]
        test_count:         测试期数
        pretrained_models:  预训练模型 {name: (model, device)}
        state:              状态字典

    返回:
        dict {test_index: {'numbers': [n1,..,n7], 'probs': [p0,..,p6]}}
    """
    predictions = {}

    try:
        # EVOLVE-BLOCK-START
        # ════════════════════════════════════════════════════════
        # 可进化的集成策略
        #
        # OpenEvolve 会修改这个区域内的代码来优化预测效果。
        # 可以修改：
        #   - 各方法权重
        #   - 权重动态调整逻辑
        #   - 概率后处理（温度缩放、平滑、锐化）
        #   - 特征驱动的权重分配
        #   - 位置级别的差异化策略
        # ════════════════════════════════════════════════════════

        # ── 集成权重定义 ──
        # 每个统计方法的基础权重
        STAT_WEIGHTS = {
            'frequency':       1.0,
            'recent_weighted': 1.3,
            'markov':          1.5,
            'hot_cold':        1.2,
            'bayesian':        1.8,
            'pattern_match':   1.4,
        }

        # 深度学习模型权重（每个模型）
        DL_MODEL_WEIGHT = 2.0

        # 温度参数：< 1.0 → 锐化概率分布，> 1.0 → 平滑
        TEMPERATURE = 0.8

        # 是否根据历史数据量动态调整权重
        ADAPTIVE_WEIGHTS = True

        # 位置7特殊处理权重放大系数
        POS7_BAYESIAN_BOOST = 1.5

        # 近期窗口大小
        LOOKBACK_WINDOW = 15

        # 马尔可夫链阶数参考窗口
        MARKOV_WEIGHT_BOOST_THRESHOLD = 100  # 数据量 > 此值时提升马尔可夫权重

        # ── 逐期预测 ──
        for test_idx in range(test_count):
            # 当前可用历史 = 训练数据 + 之前的测试预测
            # （注意：我们只用训练数据，不用测试数据 — 无数据泄露）
            current_history = list(train_data)

            # 如果已经预测了之前的期，可以把预测结果加入历史
            # （滚动预测模式）
            for prev_idx in range(test_idx):
                prev_pred = predictions.get(prev_idx)
                if prev_pred:
                    current_history.append({
                        'issue': f'pred_{prev_idx}',
                        'numbers': prev_pred['numbers']
                    })

            # ── 构建上下文特征 ──
            ctx = build_context_features(current_history, lookback=LOOKBACK_WINDOW)
            n_hist = ctx['n_history']

            # ── 动态权重调整 ──
            weights = dict(STAT_WEIGHTS)
            if ADAPTIVE_WEIGHTS:
                # 数据量少时，偏重简单方法
                if n_hist < 50:
                    weights['frequency'] *= 1.5
                    weights['bayesian'] *= 0.8
                    weights['pattern_match'] *= 0.5
                # 数据量多时，偏重复杂方法
                elif n_hist > MARKOV_WEIGHT_BOOST_THRESHOLD:
                    weights['markov'] *= 1.3
                    weights['pattern_match'] *= 1.5

            # ── 收集各方法概率分布 ──
            all_probs = []  # [(probs_list, weight)]

            # 统计方法
            stat_methods = [
                ('frequency',       stat_frequency_probs),
                ('recent_weighted', lambda h: stat_recent_weighted_probs(h, decay=0.9)),
                ('markov',          stat_markov_probs),
                ('hot_cold',        lambda h: stat_hot_cold_probs(h, window=LOOKBACK_WINDOW)),
                ('bayesian',        lambda h: stat_bayesian_probs(h, window=30)),
                ('pattern_match',   stat_pattern_match_probs),
            ]

            for name, func in stat_methods:
                try:
                    probs = func(current_history)
                    w = weights.get(name, 1.0)
                    all_probs.append((probs, w))
                except Exception:
                    pass

            # 深度学习模型
            if pretrained_models:
                for model_name, (model, device) in pretrained_models.items():
                    try:
                        probs = dl_model_probs(model, device, current_history, seq_len=5)
                        if probs is not None:
                            all_probs.append((probs, DL_MODEL_WEIGHT))
                    except Exception:
                        pass

            # ── 加权融合 ──
            if not all_probs:
                # 无可用方法，均匀分布
                fused = [np.ones(POS_RANGES[p]) / POS_RANGES[p]
                         for p in range(POS_COUNT)]
            else:
                fused = [np.zeros(POS_RANGES[p]) for p in range(POS_COUNT)]
                total_w = 0

                for probs, w in all_probs:
                    for pos in range(POS_COUNT):
                        # 位置7特殊加权
                        effective_w = w
                        if pos == 6:
                            # 贝叶斯在位置7更可靠（因为范围更大）
                            effective_w *= POS7_BAYESIAN_BOOST if w > 1.5 else 1.0

                        fused[pos] += probs[pos] * effective_w
                    total_w += w

                for pos in range(POS_COUNT):
                    fused[pos] /= (total_w + 1e-10)

            # ── 温度缩放 ──
            for pos in range(POS_COUNT):
                if TEMPERATURE != 1.0:
                    log_p = np.log(fused[pos] + 1e-10) / TEMPERATURE
                    log_p -= log_p.max()
                    fused[pos] = np.exp(log_p)
                fused[pos] /= fused[pos].sum()

            # ── 生成最终预测 ──
            pred_numbers = [int(np.argmax(fused[p])) for p in range(POS_COUNT)]

            predictions[test_idx] = {
                'numbers': pred_numbers,
                'probs': fused,
            }

        # EVOLVE-BLOCK-END

    except Exception as e:
        import traceback
        traceback.print_exc()
        # 兜底：均匀随机预测
        for test_idx in range(test_count):
            if test_idx not in predictions:
                predictions[test_idx] = {
                    'numbers': [np.random.randint(0, POS_RANGES[p])
                                for p in range(POS_COUNT)],
                    'probs': [np.ones(POS_RANGES[p]) / POS_RANGES[p]
                              for p in range(POS_COUNT)],
                }

    return predictions


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def main():
    import pandas as pd
    print("请通过 evaluator.py 调用本程序")


if __name__ == "__main__":
    main()
