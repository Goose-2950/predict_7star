"""
OpenEvolve 评估器 — 七星彩预测版

流程:
  1. 加载历史开奖数据 (CSV)
  2. 划分训练集 (85%) / 测试集 (15%)
  3. 预训练 LSTM/GRU 基础模型 (缓存到 data/)
  4. 动态加载进化后的 initial_program.py
  5. 调用 generate_x() 获取对测试期的预测
  6. 回测: 逐期比对预测 vs 实际，计算命中率
  7. 评分 0~100

评分规则:
  - 精确命中率 <= 随机基线 → 0 分
  - 精确命中率 >= 2×基线   → 60 分
  - Top3命中率额外加分 (最高40分)
  - 总分 = exact_score + top3_score，上限 100
"""

import os
import sys
import csv
import json
import time
import importlib.util
import numpy as np
from collections import Counter
from datetime import datetime

# ===================== 配置 =====================
POS_RANGES = [10, 10, 10, 10, 10, 10, 15]
POS_COUNT = 7
TRAIN_RATIO = 0.85
MAX_EVAL_TIME = 600  # 秒

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_CSV_PATH = os.path.join(_DATA_DIR, "data.csv")
_STATE_PATH = os.path.join(_DATA_DIR, "_opt_state.json")
_HISTORY_LOG = os.path.join(_DATA_DIR, "eval_history.csv")

# 随机基线 (每个位置的精确命中概率)
RANDOM_EXACT_BASELINES = [1.0 / r for r in POS_RANGES]  # ~10%, 10%, ..., 6.7%
RANDOM_EXACT_AVG = np.mean(RANDOM_EXACT_BASELINES)       # ~9.3%
RANDOM_TOP3_AVG = np.mean([min(3.0 / r, 1.0) for r in POS_RANGES])  # ~28.6%

# ===================== PyTorch =====================
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    pass


# ── 数据加载 ──

def _load_lottery_data(path=_CSV_PATH):
    """加载七星彩历史数据，返回 [{issue, numbers}, ...]"""
    data = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nums = [int(row[f'号码{i}']) for i in range(1, 8)]
            data.append({'issue': row['期号'], 'numbers': nums})
    data.sort(key=lambda x: x['issue'])
    return data


def _split_data(data, train_ratio=TRAIN_RATIO):
    """时序划分：前 train_ratio 训练，后面测试"""
    n = len(data)
    split = int(n * train_ratio)
    return data[:split], data[split:]


# ── 预训练模型 ──

if HAS_TORCH:

    class LotteryDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    class LSTMModel(nn.Module):
        def __init__(self, hidden=64, n_layers=2, dropout=0.3):
            super().__init__()
            self.emb_dim = 8
            self.embeddings = nn.ModuleList([
                nn.Embedding(POS_RANGES[p], self.emb_dim) for p in range(POS_COUNT)])
            self.lstm = nn.LSTM(
                POS_COUNT * self.emb_dim, hidden, n_layers,
                batch_first=True, dropout=dropout if n_layers > 1 else 0,
                bidirectional=True)
            self.norm = nn.LayerNorm(hidden * 2)
            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden*2, 32), nn.ReLU(),
                              nn.Dropout(dropout), nn.Linear(32, POS_RANGES[p]))
                for p in range(POS_COUNT)])

        def forward(self, x):
            xl = x.long()
            emb = torch.cat([self.embeddings[p](xl[:,:,p].clamp(0, POS_RANGES[p]-1))
                             for p in range(POS_COUNT)], dim=-1)
            out, _ = self.lstm(emb)
            out = self.dropout(self.norm(out[:, -1, :]))
            return [h(out) for h in self.heads]

    class GRUModel(nn.Module):
        def __init__(self, hidden=64, n_layers=2, dropout=0.3):
            super().__init__()
            self.emb_dim = 8
            self.embeddings = nn.ModuleList([
                nn.Embedding(POS_RANGES[p], self.emb_dim) for p in range(POS_COUNT)])
            self.gru = nn.GRU(
                POS_COUNT * self.emb_dim, hidden, n_layers,
                batch_first=True, dropout=dropout if n_layers > 1 else 0,
                bidirectional=True)
            self.norm = nn.LayerNorm(hidden * 2)
            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden*2, 32), nn.ReLU(),
                              nn.Dropout(dropout), nn.Linear(32, POS_RANGES[p]))
                for p in range(POS_COUNT)])

        def forward(self, x):
            xl = x.long()
            emb = torch.cat([self.embeddings[p](xl[:,:,p].clamp(0, POS_RANGES[p]-1))
                             for p in range(POS_COUNT)], dim=-1)
            out, _ = self.gru(emb)
            out = self.dropout(self.norm(out[:, -1, :]))
            return [h(out) for h in self.heads]

    def _make_sequences(history, seq_len=5):
        numbers = np.array([r['numbers'] for r in history])
        if len(numbers) <= seq_len:
            return np.array([]), np.array([])
        X = np.array([numbers[i:i+seq_len] for i in range(len(numbers)-seq_len)])
        y = np.array([numbers[i+seq_len] for i in range(len(numbers)-seq_len)])
        return X, y

    def _train_model(model_class, train_data, name, seq_len=5, epochs=300, lr=0.003):
        """训练并缓存模型"""
        path = os.path.join(_DATA_DIR, f"pretrained_{name}.pt")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = model_class(hidden=64, n_layers=2, dropout=0.3).to(device)

        # 尝试加载缓存
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                model.eval()
                print(f"[evaluator] Loaded cached {name} model.")
                return model, device
            except Exception:
                pass

        X, y = _make_sequences(train_data, seq_len)
        if len(X) < 5:
            return None, device

        # 数据增强
        X_aug = [X]
        y_aug = [y]
        for ai in range(2):
            rng = np.random.RandomState(42 + ai)
            Xn = X.copy().astype(np.float32)
            mask = rng.random(Xn.shape) < 0.15
            pert = rng.choice([-1, 0, 1], size=Xn.shape)
            Xn += mask * pert
            for pos in range(POS_COUNT):
                Xn[:, :, pos] = np.clip(Xn[:, :, pos], 0, POS_RANGES[pos] - 1)
            X_aug.append(Xn)
            y_aug.append(y.copy())
        X_all = np.concatenate(X_aug)
        y_all = np.concatenate(y_aug)

        dataset = LotteryDataset(X_all, y_all)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_loss = float('inf')
        best_state = None
        patience = 0

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = sum(criterion(out[p], by[:, p]) for p in range(POS_COUNT))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            avg = total_loss / len(loader)

            if avg < best_loss:
                best_loss = avg
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 40:
                break
            if (epoch + 1) % 100 == 0:
                print(f"[evaluator] {name} epoch {epoch+1}/{epochs} loss={avg:.4f}")

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        # 缓存
        os.makedirs(_DATA_DIR, exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"[evaluator] Trained & cached {name} model.")

        return model, device

    def _pretrain_all(train_data, seq_len=5):
        """预训练所有基础模型"""
        models = {}
        for name, cls in [('lstm', LSTMModel), ('gru', GRUModel)]:
            for i in range(3):
                key = f"{name}_{i}"
                torch.manual_seed(42 + i * 7)
                np.random.seed(42 + i * 7)
                m, dev = _train_model(cls, train_data, key, seq_len)
                if m is not None:
                    models[key] = (m, dev)
        return models

    def _model_predict_probs(model, device, history, seq_len=5):
        """用模型预测下一期的概率分布"""
        numbers = np.array([r['numbers'] for r in history])
        if len(numbers) < seq_len:
            return None
        seq = numbers[-seq_len:]
        x = torch.FloatTensor(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            probs = [torch.softmax(out[p], -1).cpu().numpy()[0] for p in range(POS_COUNT)]
        return probs


# ── 状态管理 ──

def _load_state():
    try:
        if os.path.exists(_STATE_PATH):
            with open(_STATE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_state(state):
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


# ── 评分逻辑 ──

def _score_predictions(predictions, test_data):
    """
    评分：对比预测 vs 实际

    predictions: dict {test_index: {
        'numbers': [n1,...,n7],          # 首选预测
        'probs': [pos0_probs, ...]       # 各位置概率分布 (可选)
    }}

    返回: (score_0_to_100, details_dict)
    """
    n_test = len(test_data)
    if n_test == 0:
        return 0.0, {"error": "no test data"}

    exact_hits = np.zeros(POS_COUNT)
    top3_hits = np.zeros(POS_COUNT)
    n_predicted = 0

    for idx in range(n_test):
        pred_info = predictions.get(idx)
        if pred_info is None:
            continue
        n_predicted += 1

        actual = test_data[idx]['numbers']
        pred_nums = pred_info.get('numbers', [])
        pred_probs = pred_info.get('probs', None)

        for pos in range(POS_COUNT):
            # 精确命中
            if pos < len(pred_nums) and pred_nums[pos] == actual[pos]:
                exact_hits[pos] += 1

            # Top3 命中 (如果有概率分布)
            if pred_probs and pos < len(pred_probs):
                top3_idx = np.argsort(pred_probs[pos])[::-1][:3]
                if actual[pos] in top3_idx:
                    top3_hits[pos] += 1
            elif pos < len(pred_nums) and pred_nums[pos] == actual[pos]:
                # 无概率信息时，精确命中也算 top3 命中
                top3_hits[pos] += 1

    if n_predicted == 0:
        return 0.0, {"error": "no predictions generated"}

    exact_rates = exact_hits / n_predicted
    top3_rates = top3_hits / n_predicted
    avg_exact = float(np.mean(exact_rates))
    avg_top3 = float(np.mean(top3_rates))

    # ── 评分公式 ──
    # Part 1: 精确命中 (0~60分)
    #   baseline (~9.3%) → 0分
    #   2× baseline (~18.6%) → 60分
    exact_target = RANDOM_EXACT_AVG * 2.0
    if avg_exact <= RANDOM_EXACT_AVG:
        exact_score = 0.0
    elif avg_exact >= exact_target:
        exact_score = 60.0
    else:
        exact_score = (avg_exact - RANDOM_EXACT_AVG) / (exact_target - RANDOM_EXACT_AVG) * 60.0

    # Part 2: Top3 命中 (0~40分)
    #   baseline (~28.6%) → 0分
    #   2× baseline (~57.2%) → 40分
    top3_target = RANDOM_TOP3_AVG * 2.0
    if avg_top3 <= RANDOM_TOP3_AVG:
        top3_score = 0.0
    elif avg_top3 >= top3_target:
        top3_score = 40.0
    else:
        top3_score = (avg_top3 - RANDOM_TOP3_AVG) / (top3_target - RANDOM_TOP3_AVG) * 40.0

    total_score = min(100.0, exact_score + top3_score)

    details = {
        "n_test": n_test,
        "n_predicted": n_predicted,
        "avg_exact_rate": round(avg_exact * 100, 2),
        "avg_top3_rate": round(avg_top3 * 100, 2),
        "random_exact_baseline": round(RANDOM_EXACT_AVG * 100, 2),
        "random_top3_baseline": round(RANDOM_TOP3_AVG * 100, 2),
        "exact_score": round(exact_score, 2),
        "top3_score": round(top3_score, 2),
        "per_pos_exact": [round(float(r) * 100, 2) for r in exact_rates],
        "per_pos_top3": [round(float(r) * 100, 2) for r in top3_rates],
    }
    return total_score, details


# ── 评估日志 ──

def _log_eval(score, details, elapsed, tag=""):
    os.makedirs(_DATA_DIR, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "score": round(score, 4),
        "elapsed_s": round(elapsed, 1),
        "tag": tag,
    }
    record.update(details)

    if not os.path.exists(_HISTORY_LOG):
        with open(_HISTORY_LOG, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            writer.writeheader()
            writer.writerow(record)
    else:
        with open(_HISTORY_LOG, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            writer.writerow(record)


# ── 主评估入口 ──

def evaluate(path_user_py: str) -> dict:
    """
    OpenEvolve 评估器入口。

    参数:
        path_user_py: 进化后的 initial_program.py 路径

    返回:
        dict: {"combined_score": float(0-100), ...}
    """
    t0 = time.time()

    try:
        # ──── 1. 加载数据 ────
        if not os.path.exists(_CSV_PATH):
            return {"combined_score": 0.0, "error": f"CSV not found: {_CSV_PATH}"}

        all_data = _load_lottery_data(_CSV_PATH)
        if len(all_data) < 20:
            return {"combined_score": 0.0, "error": "insufficient data"}

        train_data, test_data = _split_data(all_data)
        print(f"[evaluator] Data: {len(all_data)} total, "
              f"{len(train_data)} train, {len(test_data)} test")

        # ──── 2. 预训练基础模型 ────
        pretrained_models = {}
        if HAS_TORCH:
            pretrained_models = _pretrain_all(train_data, seq_len=5)
            print(f"[evaluator] Pretrained models: {list(pretrained_models.keys())}")
        else:
            print("[evaluator] PyTorch not available, no pretrained models.")

        # ──── 3. 加载状态 ────
        state = _load_state()
        state["pos_ranges"] = POS_RANGES
        state["pos_count"] = POS_COUNT
        state["n_train"] = len(train_data)
        state["n_test"] = len(test_data)

        if time.time() - t0 > MAX_EVAL_TIME:
            return {"combined_score": 0.0, "error": "timeout loading models"}

        # ──── 4. 动态加载进化后的程序 ────
        spec = importlib.util.spec_from_file_location("evolved_prog", path_user_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # ──── 5. 调用 generate_x() ────
        result = mod.generate_x(
            train_data=train_data,
            test_count=len(test_data),
            pretrained_models=pretrained_models,
            state=state,
        )

        if time.time() - t0 > MAX_EVAL_TIME:
            return {"combined_score": 0.0, "error": "timeout in generate_x()"}

        # ──── 6. 解析返回值 ────
        # 期望格式: dict {test_index: {'numbers': [...], 'probs': [...]}}
        # 或 list[dict] 按顺序对应 test_data
        predictions = {}

        if isinstance(result, dict):
            predictions = result
        elif isinstance(result, (list, tuple)):
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    predictions[i] = item
                elif isinstance(item, (list, tuple)) and len(item) == POS_COUNT:
                    predictions[i] = {'numbers': list(item)}
        else:
            return {"combined_score": 0.0, "error": "invalid return format"}

        print(f"[evaluator] Got {len(predictions)} predictions for {len(test_data)} test periods")

        # ──── 7. 评分 ────
        score, details = _score_predictions(predictions, test_data)

        elapsed = time.time() - t0
        print(f"[evaluator] Score: {score:.2f}/100 "
              f"(exact={details['avg_exact_rate']}%, "
              f"top3={details['avg_top3_rate']}%)")

        # ──── 8. 保存状态 & 日志 ────
        state["eval_count"] = state.get("eval_count", 0) + 1
        state["best_score"] = max(state.get("best_score", 0), score)
        _save_state(state)
        _log_eval(score, details, elapsed)

        return {
            "combined_score": float(max(0.0, min(100.0, score))),
            **details,
            "elapsed_s": round(elapsed, 1),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"combined_score": 0.0, "error": str(e)}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "initial_program.py")
    result = evaluate(os.path.abspath(path))
    print(json.dumps(result, indent=2, ensure_ascii=False))
