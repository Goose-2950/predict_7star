"""
7星彩号码智能预测系统 v4.0
═══════════════════════════
12种算法集成预测:
  6种统计 + LSTM + GRU + GAN + 强化学习 + 贝叶斯 + 随机森林

★ 位置7范围 0~14 | 训练/验证/测试划分 | 回测评估

依赖:
  pip install numpy torch scikit-learn
"""

import csv
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import warnings
import time
import os
import json

warnings.filterwarnings('ignore')

# ============================================================
# 全局常量
# ============================================================
POS_RANGES = [10, 10, 10, 10, 10, 10, 15]
POS_COUNT = 7

# ============================================================
# 依赖检测
# ============================================================
HAS_TORCH = False
HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    pass

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    pass

if HAS_TORCH and torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
#  数据加载
# ============================================================
def load_data_from_csv(filename='7星彩_历史开奖号码.csv'):
    data = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nums = [int(row[f'号码{i}']) for i in range(1, 8)]
            data.append({
                'issue': row['期号'],
                'date': row.get('开奖日期', ''),
                'numbers': nums,
            })
    return data

def load_data_from_json(filename='raw_api_response.json'):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for item in json_data['data']:
            front_nums = [int(x) for x in item['frontWinningNum'].split()]
            back_num = [int(item['backWinningNum'])]
            nums = front_nums + back_num
            data.append({
                'issue': item['issue'],
                'date': item['openTime'],
                'numbers': nums,
            })
    return data

def load_data(csv_filename='7星彩_历史开奖号码.csv', json_filename='raw_api_response.json'):
    # 加载 CSV 数据
    csv_data = load_data_from_csv(csv_filename)
    
    # 加载 JSON 数据
    json_data = load_data_from_json(json_filename)
    
    # 合并数据
    all_data = csv_data + json_data
    
    # 去重和排序
    seen_issues = set()
    unique_data = []
    for rec in all_data:
        if rec['issue'] not in seen_issues:
            seen_issues.add(rec['issue'])
            unique_data.append(rec)
    
    unique_data.sort(key=lambda x: x['issue'])

    errors = 0
    for rec in unique_data:
        for pos in range(POS_COUNT):
            if not (0 <= rec['numbers'][pos] < POS_RANGES[pos]):
                errors += 1
    if errors:
        print(f"  ⚠ 数据验证: {errors} 个范围异常")
    else:
        print(f"  OK 数据验证: 全部 {len(unique_data)} 条范围正确")
    return unique_data


def get_builtin_data():
    raw = [
        ("26018", [2, 9, 3, 6, 5, 7, 2]),
        ("26019", [6, 0, 8, 8, 2, 7, 1]),
        ("26020", [5, 5, 0, 9, 5, 6, 4]),
        ("26021", [6, 4, 3, 5, 6, 3, 2]),
        ("26022", [7, 9, 9, 1, 0, 2, 3]),
        ("26023", [2, 8, 3, 8, 4, 8, 11]),
        ("26024", [5, 4, 4, 9, 0, 3, 14]),
        ("26025", [8, 4, 0, 5, 5, 3, 1]),
        ("26026", [2, 6, 9, 8, 3, 8, 12]),
        ("26027", [8, 7, 7, 3, 6, 0, 9]),
        ("26028", [7, 9, 8, 0, 1, 1, 6]),
        ("26029", [1, 4, 0, 4, 3, 2, 13]),
        ("26030", [1, 9, 8, 1, 5, 9, 1]),
        ("26031", [6, 8, 9, 5, 2, 2, 10]),
        ("26032", [3, 1, 6, 4, 4, 5, 1]),
        ("26033", [1, 8, 9, 1, 9, 3, 1]),
        ("26034", [3, 9, 9, 1, 5, 3, 5]),
    ]
    return [{'issue': r[0], 'date': '', 'numbers': r[1]} for r in raw]


def split_dataset(history, train_ratio=0.70, val_ratio=0.15):
    n = len(history)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return history[:t1], history[t1:t2], history[t2:]


# ############################################################
#  工具函数
# ############################################################

def build_features(history, idx, lookback=10):
    """
    为第idx期构造特征向量（供随机森林、RL等使用）
    特征包括: 近N期每位置频率、差值、奇偶比、和值等
    """
    start = max(0, idx - lookback)
    recent = history[start:idx]
    if not recent:
        return None

    features = []
    for pos in range(POS_COUNT):
        n_vals = POS_RANGES[pos]
        nums = [r['numbers'][pos] for r in recent]
        counter = Counter(nums)

        # 各数字频率
        freq = [counter.get(v, 0) / len(nums) for v in range(n_vals)]
        features.extend(freq)

        # 最近一期的值 (归一化)
        features.append(recent[-1]['numbers'][pos] / (n_vals - 1))

        # 最近差值
        if len(nums) >= 2:
            features.append((nums[-1] - nums[-2]) / (n_vals - 1))
        else:
            features.append(0.0)

        # 均值、标准差
        features.append(np.mean(nums) / (n_vals - 1))
        features.append(np.std(nums) / (n_vals - 1) if len(nums) > 1 else 0.0)

    # 全局特征
    sums = [sum(r['numbers']) for r in recent]
    features.append(np.mean(sums) / 50)
    features.append(np.std(sums) / 20 if len(sums) > 1 else 0.0)

    # 奇偶比
    last_nums = recent[-1]['numbers']
    odd_ratio = sum(1 for x in last_nums if x % 2 == 1) / POS_COUNT
    features.append(odd_ratio)

    return np.array(features, dtype=np.float32)


def get_device():
    """智能选择设备"""
    device = torch.device('cpu')
    if torch.cuda.is_available():
        try:
            free_mem = torch.cuda.mem_get_info()[0] / 1024 ** 2
            total_mem = torch.cuda.mem_get_info()[1] / 1024 ** 2
            if free_mem > 256:
                device = torch.device('cuda')
                return device
        except Exception:
            pass
    return device


# ############################################################
#  第一部分：6种统计方法（与 v3.1 相同）
# ############################################################

def method_frequency(history, n_predict=5):
    nums_by_pos = [[] for _ in range(POS_COUNT)]
    for rec in history:
        for i in range(POS_COUNT):
            nums_by_pos[i].append(rec['numbers'][i])
    predictions = []
    for _ in range(n_predict):
        pred = []
        for pos in range(POS_COUNT):
            n_vals = POS_RANGES[pos]
            counter = Counter(nums_by_pos[pos])
            values = list(range(n_vals))
            weights = [counter.get(v, 0) + 0.1 for v in values]
            total = sum(weights)
            probs = [w / total for w in weights]
            pred.append(int(np.random.choice(values, p=probs)))
        predictions.append(pred)
    return predictions


def method_hot_cold(history, n_predict=5, window=10):
    recent = history[-window:]
    nums_by_pos = [[] for _ in range(POS_COUNT)]
    for rec in recent:
        for i in range(POS_COUNT):
            nums_by_pos[i].append(rec['numbers'][i])
    predictions = []
    for _ in range(n_predict):
        pred = []
        for pos in range(POS_COUNT):
            n_vals = POS_RANGES[pos]
            counter = Counter(nums_by_pos[pos])
            values = list(range(n_vals))
            weights = []
            for v in values:
                freq = counter.get(v, 0)
                if freq >= 2:
                    weights.append(freq * 2.0)
                elif freq == 0:
                    weights.append(1.5)
                else:
                    weights.append(1.0)
            total = sum(weights)
            probs = [w / total for w in weights]
            pred.append(int(np.random.choice(values, p=probs)))
        predictions.append(pred)
    return predictions


def method_markov(history, n_predict=5):
    transition = []
    for pos in range(POS_COUNT):
        n_vals = POS_RANGES[pos]
        matrix = np.ones((n_vals, n_vals)) * 0.01
        for j in range(1, len(history)):
            prev = history[j - 1]['numbers'][pos]
            curr = history[j]['numbers'][pos]
            matrix[prev][curr] += 1
        for row in range(n_vals):
            matrix[row] /= matrix[row].sum()
        transition.append(matrix)
    last_numbers = history[-1]['numbers']
    predictions = []
    current = list(last_numbers)
    for _ in range(n_predict):
        pred = []
        for pos in range(POS_COUNT):
            probs = transition[pos][current[pos]]
            pred.append(int(np.random.choice(POS_RANGES[pos], p=probs)))
        predictions.append(pred)
        current = list(pred)
    return predictions


def method_weighted_recent(history, n_predict=5):
    n = len(history)
    decay = 0.9
    predictions = []
    for _ in range(n_predict):
        pred = []
        for pos in range(POS_COUNT):
            n_vals = POS_RANGES[pos]
            score = np.zeros(n_vals)
            for j in range(n):
                num = history[j]['numbers'][pos]
                score[num] += decay ** (n - 1 - j)
            probs = score / score.sum()
            pred.append(int(np.random.choice(n_vals, p=probs)))
        predictions.append(pred)
    return predictions


def method_delta_trend(history, n_predict=5, window=8):
    recent = history[-window:]
    predictions = []
    last_nums = history[-1]['numbers']
    for _ in range(n_predict):
        pred = []
        for pos in range(POS_COUNT):
            n_vals = POS_RANGES[pos]
            deltas = []
            for j in range(1, len(recent)):
                deltas.append(recent[j]['numbers'][pos] - recent[j - 1]['numbers'][pos])
            if not deltas:
                pred.append(last_nums[pos])
                continue
            weights = [0.5 ** (len(deltas) - 1 - i) for i in range(len(deltas))]
            avg_delta = sum(d * w for d, w in zip(deltas, weights)) / sum(weights)
            next_val = int(round(last_nums[pos] + avg_delta + np.random.normal(0, 1.5)))
            next_val = next_val % n_vals
            if next_val < 0:
                next_val += n_vals
            pred.append(next_val)
        predictions.append(pred)
        last_nums = pred
    return predictions


def method_pattern_match(history, n_predict=5, match_len=3):
    if len(history) < match_len + 1:
        return method_frequency(history, n_predict)
    query = [rec['numbers'] for rec in history[-match_len:]]
    similarities = []
    for i in range(len(history) - match_len - 1):
        segment = [history[i + j]['numbers'] for j in range(match_len)]
        score = 0
        for t in range(match_len):
            for pos in range(POS_COUNT):
                if segment[t][pos] == query[t][pos]:
                    score += 1
                elif abs(segment[t][pos] - query[t][pos]) <= 1:
                    score += 0.5
        similarities.append((score, i + match_len))
    similarities.sort(key=lambda x: x[0], reverse=True)
    predictions = []
    for step in range(n_predict):
        pos_scores = [np.zeros(POS_RANGES[p]) for p in range(POS_COUNT)]
        for sim_score, next_idx in similarities[:10]:
            if next_idx + step < len(history):
                for pos in range(POS_COUNT):
                    num = history[next_idx + step]['numbers'][pos]
                    pos_scores[pos][num] += sim_score + 0.01
        pred = []
        for pos in range(POS_COUNT):
            scores = pos_scores[pos]
            if scores.sum() == 0:
                pred.append(np.random.randint(0, POS_RANGES[pos]))
            else:
                pred.append(int(np.random.choice(POS_RANGES[pos], p=scores / scores.sum())))
        predictions.append(pred)
    return predictions


# ############################################################
#  第二部分：PyTorch 通用基础设施
# ############################################################

if HAS_TORCH:

    class LotteryDataset(Dataset):
        def __init__(self, sequences, targets):
            self.X = torch.FloatTensor(sequences)
            self.y = torch.LongTensor(targets)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    def make_sequences(history_list, seq_len=5):
        numbers = np.array([rec['numbers'] for rec in history_list])
        X, y = [], []
        for i in range(len(numbers) - seq_len):
            X.append(numbers[i:i + seq_len])
            y.append(numbers[i + seq_len])
        if not X:
            return np.array([]), np.array([])
        return np.array(X), np.array(y)

    def evaluate_model(model, device, X, y):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        dataset = LotteryDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        total_loss = 0
        pos_correct = np.zeros(POS_COUNT)
        pos_top3_correct = np.zeros(POS_COUNT)
        total_samples = 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = sum(criterion(outputs[p], batch_y[:, p]) for p in range(POS_COUNT))
                total_loss += loss.item() * batch_X.size(0)
                for p in range(POS_COUNT):
                    preds = torch.argmax(outputs[p], dim=-1)
                    pos_correct[p] += (preds == batch_y[:, p]).sum().item()
                    k = min(3, POS_RANGES[p])
                    top3 = torch.topk(outputs[p], k, dim=-1).indices
                    for ri in range(batch_X.size(0)):
                        if batch_y[ri, p] in top3[ri]:
                            pos_top3_correct[p] += 1
                total_samples += batch_X.size(0)
        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss, pos_correct / max(total_samples, 1), pos_top3_correct / max(total_samples, 1)


# ############################################################
#  模型 7：LSTM（保留原有，略微优化）
# ############################################################

if HAS_TORCH:

    class LSTMModel(nn.Module):
        def __init__(self, hidden_size=64, num_layers=2, dropout=0.5):
            super().__init__()
            self.emb_dim = 8
            self.embeddings = nn.ModuleList([
                nn.Embedding(POS_RANGES[p], self.emb_dim) for p in range(POS_COUNT)
            ])
            self.lstm = nn.LSTM(
                input_size=POS_COUNT * self.emb_dim,
                hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_size * 2)
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size * 2, 32), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(32, POS_RANGES[p]),
                ) for p in range(POS_COUNT)
            ])

        def forward(self, x):
            bs, sl, _ = x.shape
            xl = x.long()
            emb = torch.cat([
                self.embeddings[p](xl[:, :, p].clamp(0, POS_RANGES[p] - 1))
                for p in range(POS_COUNT)
            ], dim=-1)
            out, _ = self.lstm(emb)
            out = self.dropout(self.norm(out[:, -1, :]))
            return [head(out) for head in self.heads]


# ############################################################
#  模型 8：GRU（新增）
# ############################################################

if HAS_TORCH:

    class GRUModel(nn.Module):
        """
        GRU 替代 LSTM
        ─────────────
        参数量更少，训练更快，对小数据集可能效果更好
        """
        def __init__(self, hidden_size=64, num_layers=2, dropout=0.5):
            super().__init__()
            self.emb_dim = 8
            self.embeddings = nn.ModuleList([
                nn.Embedding(POS_RANGES[p], self.emb_dim) for p in range(POS_COUNT)
            ])
            self.gru = nn.GRU(
                input_size=POS_COUNT * self.emb_dim,
                hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_size * 2)
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size * 2, 32), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(32, POS_RANGES[p]),
                ) for p in range(POS_COUNT)
            ])

        def forward(self, x):
            bs, sl, _ = x.shape
            xl = x.long()
            emb = torch.cat([
                self.embeddings[p](xl[:, :, p].clamp(0, POS_RANGES[p] - 1))
                for p in range(POS_COUNT)
            ], dim=-1)
            out, _ = self.gru(emb)
            out = self.dropout(self.norm(out[:, -1, :]))
            return [head(out) for head in self.heads]


# ############################################################
#  模型 9：GAN — 生成对抗网络（新增）
# ############################################################

if HAS_TORCH:

    class NumberRelationEncoder(nn.Module):
        """
        号码关联编码器（简化版 GNN 思想）
        ─────────────────────────────────
        对7个位置的 embedding 做交叉注意力，
        学习号码间的拓扑关系（连号、尾数匹配等）
        """
        def __init__(self, emb_dim=8, n_heads=2):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                embed_dim=emb_dim, num_heads=n_heads, batch_first=True
            )
            self.norm = nn.LayerNorm(emb_dim)

        def forward(self, pos_embeddings):
            # pos_embeddings: (batch, 7, emb_dim)
            attn_out, _ = self.attention(pos_embeddings, pos_embeddings, pos_embeddings)
            return self.norm(pos_embeddings + attn_out)

    class Generator(nn.Module):
        """
        GAN 生成器
        ──────────
        输入: 噪声 z + 条件向量（最近几期的统计特征）
        输出: 7个位置各自的概率分布
        """
        def __init__(self, noise_dim=32, cond_dim=64, hidden=128):
            super().__init__()
            self.emb_dim = 8

            # 条件编码器（将历史序列编码为条件向量）
            self.cond_embeddings = nn.ModuleList([
                nn.Embedding(POS_RANGES[p], self.emb_dim) for p in range(POS_COUNT)
            ])
            self.cond_gru = nn.GRU(
                POS_COUNT * self.emb_dim, cond_dim,
                batch_first=True, num_layers=1
            )

            # 号码关联编码器（GNN思想）
            self.relation_encoder = NumberRelationEncoder(self.emb_dim)

            # 生成网络
            self.net = nn.Sequential(
                nn.Linear(noise_dim + cond_dim, hidden),
                nn.LayerNorm(hidden),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.LeakyReLU(0.2),
            )

            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden, 32), nn.ReLU(),
                    nn.Linear(32, POS_RANGES[p]),
                ) for p in range(POS_COUNT)
            ])

        def encode_condition(self, seq):
            """编码历史序列为条件向量"""
            xl = seq.long()
            emb_list = []
            for p in range(POS_COUNT):
                emb_list.append(
                    self.cond_embeddings[p](xl[:, :, p].clamp(0, POS_RANGES[p] - 1))
                )

            # 关系编码 (对最后一个时间步)
            last_embs = torch.stack([e[:, -1, :] for e in emb_list], dim=1)  # (B, 7, emb)
            rel_embs = self.relation_encoder(last_embs)  # (B, 7, emb)

            # GRU 时序编码
            x = torch.cat(emb_list, dim=-1)
            _, h = self.cond_gru(x)
            return h[-1]  # (batch, cond_dim)

        def forward(self, z, seq):
            cond = self.encode_condition(seq)
            x = torch.cat([z, cond], dim=-1)
            x = self.net(x)
            return [head(x) for head in self.heads]

    class Discriminator(nn.Module):
        """
        GAN 判别器
        ──────────
        输入: 7个位置的号码（one-hot）+ 条件
        输出: 真/假概率
        """
        def __init__(self, cond_dim=64, hidden=128):
            super().__init__()
            total_dim = sum(POS_RANGES)  # 10*6 + 15 = 75

            self.cond_emb = nn.ModuleList([
                nn.Embedding(POS_RANGES[p], 8) for p in range(POS_COUNT)
            ])
            self.cond_gru = nn.GRU(POS_COUNT * 8, cond_dim, batch_first=True)

            self.net = nn.Sequential(
                nn.Linear(total_dim + cond_dim, hidden),
                nn.LayerNorm(hidden),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5),
                nn.Linear(hidden, hidden // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, numbers, seq):
            # numbers: list of 7 tensors, each (batch, n_vals) — soft one-hot
            x_nums = torch.cat(numbers, dim=-1)  # (batch, 75)

            xl = seq.long()
            embs = torch.cat([
                self.cond_emb[p](xl[:, :, p].clamp(0, POS_RANGES[p] - 1))
                for p in range(POS_COUNT)
            ], dim=-1)
            _, h = self.cond_gru(embs)
            cond = h[-1]

            return self.net(torch.cat([x_nums, cond], dim=-1))

    def train_gan(history, seq_len=5, epochs=200, verbose=True, early_stop=True, log_file=None):
        """训练条件 GAN"""
        X, y = make_sequences(history, seq_len)
        if len(X) < 5:
            return None, None, None

        # 增强
        X_all = np.tile(X, (3, 1, 1))
        y_all = np.tile(y, (3, 1))

        dataset = LotteryDataset(X_all, y_all)
        loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)

        device = get_device()
        noise_dim = 32

        gen = Generator(noise_dim=noise_dim, cond_dim=64, hidden=128).to(device)
        disc = Discriminator(cond_dim=64, hidden=128).to(device)

        opt_g = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(disc.parameters(), lr=0.001, betas=(0.5, 0.999))

        best_g_loss = float('inf')
        patience_counter = 0

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('epoch,g_loss,d_loss\n')

        for epoch in range(epochs):
            g_loss_sum, d_loss_sum = 0, 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                bs = batch_X.size(0)

                # ---- 构造真实 soft one-hot ----
                real_onehot = []
                for p in range(POS_COUNT):
                    oh = torch.zeros(bs, POS_RANGES[p], device=device)
                    oh.scatter_(1, batch_y[:, p:p + 1], 1.0)
                    # 加点噪声避免判别器太强
                    oh = oh * 0.9 + 0.1 / POS_RANGES[p]
                    real_onehot.append(oh)

                # ---- 训练判别器 ----
                z = torch.randn(bs, noise_dim, device=device)
                fake_logits = gen(z, batch_X)
                fake_onehot = [torch.softmax(fl, dim=-1).detach() for fl in fake_logits]

                d_real = disc(real_onehot, batch_X)
                d_fake = disc(fake_onehot, batch_X)

                d_loss = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) +
                          F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))) / 2

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                d_loss_sum += d_loss.item()

                # ---- 训练生成器 ----
                z = torch.randn(bs, noise_dim, device=device)
                fake_logits = gen(z, batch_X)
                fake_onehot_g = [torch.softmax(fl, dim=-1) for fl in fake_logits]

                d_fake_g = disc(fake_onehot_g, batch_X)
                g_adv_loss = F.binary_cross_entropy_with_logits(d_fake_g, torch.ones_like(d_fake_g))

                # 辅助分类损失（让生成器也学习预测正确号码）
                ce_loss = sum(
                    F.cross_entropy(fake_logits[p], batch_y[:, p])
                    for p in range(POS_COUNT)
                )
                g_loss = g_adv_loss + ce_loss * 0.5

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                g_loss_sum += g_loss.item()

            avg_g_loss = g_loss_sum / len(loader)
            avg_d_loss = d_loss_sum / len(loader)

            if early_stop:
                if avg_g_loss < best_g_loss:
                    best_g_loss = avg_g_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 40:
                    if verbose:
                        print(f"      GAN 早停 Epoch {epoch + 1}")
                    break

            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{epoch + 1},{avg_g_loss:.6f},{avg_d_loss:.6f}\n")

            if verbose:
                print(f"      GAN Epoch {epoch + 1:>3}/{epochs}"
                      f"  G_loss: {avg_g_loss:.4f}  D_loss: {avg_d_loss:.4f}")

        gen.eval()
        return gen, device, noise_dim

    def predict_gan(gen, device, noise_dim, history, seq_len=5,
                    n_predict=5, n_samples=50):
        """GAN 预测：生成多个样本后统计"""
        if gen is None:
            return None

        numbers = np.array([rec['numbers'] for rec in history])
        current_seq = numbers[-seq_len:].copy()
        predictions = []

        with torch.no_grad():
            for _ in range(n_predict):
                x = torch.FloatTensor(current_seq).unsqueeze(0).repeat(n_samples, 1, 1).to(device)
                z = torch.randn(n_samples, noise_dim, device=device)
                outputs = gen(z, x)

                # 统计所有样本的概率
                probs_list = []
                pred = []
                for p in range(POS_COUNT):
                    prob = torch.softmax(outputs[p], dim=-1).mean(dim=0).cpu().numpy()
                    probs_list.append(prob)
                    pred.append(int(np.argmax(prob)))

                predictions.append({'numbers': pred, 'probs': probs_list})
                new_row = np.array(pred).reshape(1, POS_COUNT)
                current_seq = np.concatenate([current_seq[1:], new_row], axis=0)

        return predictions


# ############################################################
#  模型 10：强化学习 — Policy Gradient（新增）
# ############################################################

if HAS_TORCH:

    class PolicyNetwork(nn.Module):
        """
        策略网络（Contextual Bandit）
        ────────────────────────────
        状态: 历史特征向量
        动作: 选择每个位置的数字
        奖励: 与实际号码的匹配程度
        """
        def __init__(self, state_dim, hidden=128):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
            )
            self.heads = nn.ModuleList([
                nn.Linear(hidden, POS_RANGES[p]) for p in range(POS_COUNT)
            ])

        def forward(self, state):
            x = self.shared(state)
            return [F.log_softmax(head(x), dim=-1) for head in self.heads]

        def get_action_probs(self, state):
            x = self.shared(state)
            return [F.softmax(head(x), dim=-1) for head in self.heads]

    def train_rl_policy(history, lookback=10, epochs=300, verbose=True, early_stop=True, log_file=None):
        """
        训练强化学习策略
        使用 REINFORCE 算法 + baseline
        """
        # 构建 (state, action=actual_number) 对
        states = []
        targets = []
        for i in range(lookback + 1, len(history)):
            feat = build_features(history, i, lookback)
            if feat is not None:
                states.append(feat)
                targets.append(history[i]['numbers'])

        if len(states) < 10:
            if verbose:
                print("      ⚠ RL 训练样本不足")
            return None, None, None

        states = np.array(states)
        targets = np.array(targets)
        state_dim = states.shape[1]

        device = get_device()
        policy = PolicyNetwork(state_dim, hidden=128).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.002)

        # 预训练：先用监督学习初始化策略
        states_t = torch.FloatTensor(states).to(device)
        targets_t = torch.LongTensor(targets).to(device)

        best_loss = float('inf')
        patience_counter = 0

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('epoch,total_loss,supervised_loss,rl_loss,avg_reward\n')

        for epoch in range(epochs):
            log_probs = policy(states_t)
            loss = sum(
                F.nll_loss(log_probs[p], targets_t[:, p]) for p in range(POS_COUNT)
            )

            # REINFORCE 奖励增强
            with torch.no_grad():
                probs = policy.get_action_probs(states_t)
                # 计算每个样本的"奖励" = 正确位置数
                rewards = torch.zeros(len(states), device=device)
                for p in range(POS_COUNT):
                    pred = torch.argmax(probs[p], dim=-1)
                    rewards += (pred == targets_t[:, p]).float()
                # baseline
                baseline = rewards.mean()
                advantage = rewards - baseline

            # 加权损失
            rl_loss = 0
            for p in range(POS_COUNT):
                action_log_prob = log_probs[p].gather(1, targets_t[:, p:p + 1]).squeeze()
                rl_loss -= (action_log_prob * advantage).mean() * 0.1

            total_loss = loss + rl_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if early_stop:
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 40:
                    if verbose:
                        print(f"      RL 早停 Epoch {epoch + 1}")
                    break

            avg_reward = rewards.mean().item()
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{epoch + 1},{total_loss.item():.6f},{loss.item():.6f},{rl_loss:.6f},{avg_reward:.4f}\n")

            if verbose:
                print(f"      RL Epoch {epoch + 1:>3}/{epochs}"
                      f"  Loss: {loss.item():.4f}  Avg Reward: {avg_reward:.2f}/7")

        policy.eval()
        return policy, device, state_dim

    def predict_rl(policy, device, history, lookback=10, n_predict=5):
        """RL 策略预测"""
        if policy is None:
            return None

        predictions = []
        current_history = list(history)

        for _ in range(n_predict):
            feat = build_features(current_history, len(current_history), lookback)
            if feat is None:
                break

            state = torch.FloatTensor(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = policy.get_action_probs(state)

            pred = []
            probs_list = []
            for p in range(POS_COUNT):
                prob = probs[p].cpu().numpy()[0]
                probs_list.append(prob)
                pred.append(int(np.argmax(prob)))

            predictions.append({'numbers': pred, 'probs': probs_list})
            # 追加预测结果用于滚动预测
            current_history.append({'numbers': pred, 'issue': '', 'date': ''})

        return predictions


# ############################################################
#  模型 11：贝叶斯概率模型（新增）
# ############################################################

def method_bayesian(history, n_predict=5, window=30):
    """
    贝叶斯 Dirichlet-Multinomial 模型
    ──────────────────────────────────
    先验: Dirichlet(alpha) — 均匀先验
    似然: Multinomial（观测频率）
    后验: Dirichlet(alpha + counts)

    根据近期数据权重更高来更新后验，实时调整概率分布
    """
    predictions = []

    for _ in range(n_predict):
        pred = []
        for pos in range(POS_COUNT):
            n_vals = POS_RANGES[pos]

            # 先验: 弱均匀 Dirichlet
            alpha_prior = np.ones(n_vals) * 1.0

            # 观测（全部历史 + 近期加权）
            global_counts = np.zeros(n_vals)
            for rec in history:
                global_counts[rec['numbers'][pos]] += 1

            # 近期数据更高权重
            recent = history[-window:]
            recent_counts = np.zeros(n_vals)
            for j, rec in enumerate(recent):
                # 越近权重越大
                w = 1.0 + j / len(recent) * 2.0
                recent_counts[rec['numbers'][pos]] += w

            # 后验 = 先验 + 加权观测
            alpha_post = alpha_prior + global_counts * 0.3 + recent_counts * 0.7

            # 从后验 Dirichlet 采样
            probs = np.random.dirichlet(alpha_post)
            pred.append(int(np.random.choice(n_vals, p=probs)))

        predictions.append(pred)

    return predictions


def method_bayesian_probs(history, window=30):
    """
    贝叶斯模型 — 返回各位置概率分布（用于集成投票）
    """
    probs_list = []
    for pos in range(POS_COUNT):
        n_vals = POS_RANGES[pos]
        alpha_prior = np.ones(n_vals) * 1.0

        global_counts = np.zeros(n_vals)
        for rec in history:
            global_counts[rec['numbers'][pos]] += 1

        recent = history[-window:]
        recent_counts = np.zeros(n_vals)
        for j, rec in enumerate(recent):
            w = 1.0 + j / len(recent) * 2.0
            recent_counts[rec['numbers'][pos]] += w

        alpha_post = alpha_prior + global_counts * 0.3 + recent_counts * 0.7

        # 后验均值作为概率估计（比采样更稳定）
        probs = alpha_post / alpha_post.sum()
        probs_list.append(probs)

    return probs_list


# ############################################################
#  模型 12：随机森林（新增）
# ############################################################

def method_random_forest(history, n_predict=5, lookback=10):
    """
    随机森林分类器
    ──────────────
    对每个位置训练独立的 RandomForestClassifier
    特征: build_features() 生成的滚动统计特征
    """
    if not HAS_SKLEARN:
        return method_frequency(history, n_predict)

    # 构建训练数据
    X_list, y_list = [], []
    for i in range(lookback + 1, len(history)):
        feat = build_features(history, i, lookback)
        if feat is not None:
            X_list.append(feat)
            y_list.append(history[i]['numbers'])

    if len(X_list) < 20:
        return method_frequency(history, n_predict)

    X = np.array(X_list)
    y = np.array(y_list)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 为每个位置训练一个随机森林
    models = []
    for pos in range(POS_COUNT):
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42 + pos,
            n_jobs=-1,
        )
        rf.fit(X_scaled, y[:, pos])
        models.append(rf)

    # 预测
    predictions = []
    current_history = list(history)
    for _ in range(n_predict):
        feat = build_features(current_history, len(current_history), lookback)
        if feat is None:
            break
        feat_scaled = scaler.transform(feat.reshape(1, -1))

        pred = []
        for pos in range(POS_COUNT):
            pred_val = models[pos].predict(feat_scaled)[0]
            pred.append(int(pred_val))

        predictions.append(pred)
        current_history.append({'numbers': pred, 'issue': '', 'date': ''})

    return predictions


def method_random_forest_probs(history, lookback=10):
    """随机森林 — 返回概率分布"""
    if not HAS_SKLEARN:
        return None

    X_list, y_list = [], []
    for i in range(lookback + 1, len(history)):
        feat = build_features(history, i, lookback)
        if feat is not None:
            X_list.append(feat)
            y_list.append(history[i]['numbers'])

    if len(X_list) < 20:
        return None

    X = np.array(X_list)
    y = np.array(y_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feat = build_features(history, len(history), lookback)
    if feat is None:
        return None
    feat_scaled = scaler.transform(feat.reshape(1, -1))

    probs_list = []
    for pos in range(POS_COUNT):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            class_weight='balanced', random_state=42 + pos, n_jobs=-1,
        )
        rf.fit(X_scaled, y[:, pos])
        prob = np.zeros(POS_RANGES[pos])
        if hasattr(rf, 'predict_proba'):
            p = rf.predict_proba(feat_scaled)[0]
            classes = rf.classes_
            for ci, cls in enumerate(classes):
                if 0 <= cls < POS_RANGES[pos]:
                    prob[cls] = p[ci]
        if prob.sum() == 0:
            prob = np.ones(POS_RANGES[pos]) / POS_RANGES[pos]
        else:
            prob /= prob.sum()
        probs_list.append(prob)

    return probs_list


# ############################################################
#  通用 RNN 训练函数（LSTM / GRU 共用）
# ############################################################

if HAS_TORCH:

    def train_rnn_model(model_class, train_data, val_data, seq_len=5,
                        epochs=300, lr=0.001, verbose=True,
                        model_name="RNN", model_id=1, early_stop=True,
                        log_file=None):
        """通用 RNN 训练（LSTM 和 GRU 共用）"""
        X_train, y_train = make_sequences(train_data, seq_len)
        X_val, y_val = make_sequences(val_data, seq_len) if val_data else (np.array([]), np.array([]))

        if len(X_train) < 3:
            if verbose:
                print(f"    ⚠ {model_name} 训练样本不足")
            return None, None

        X_aug = np.tile(X_train, (5, 1, 1))
        y_aug = np.tile(y_train, (5, 1))
        train_dataset = LotteryDataset(X_aug, y_aug)
        train_loader = DataLoader(train_dataset, batch_size=min(64, len(train_dataset)), shuffle=True)

        device = get_device() if model_id == 1 else (
            torch.device('cuda') if torch.cuda.is_available()
            and torch.cuda.mem_get_info()[0] / 1024 ** 2 > 256
            else torch.device('cpu')
        )

        model = model_class(hidden_size=64, num_layers=2, dropout=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('epoch,train_loss,val_loss,val_acc\n')

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                outputs = model(bx)
                loss = sum(criterion(outputs[p], by[:, p]) for p in range(POS_COUNT))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            avg_tl = train_loss / len(train_loader)

            val_loss = -1
            val_acc = -1
            if len(X_val) > 0:
                vl, pos_acc, _ = evaluate_model(model, device, X_val, y_val)
                val_loss = vl
                val_acc = np.mean(pos_acc)
                monitor = vl
            else:
                monitor = avg_tl

            if early_stop:
                if monitor < best_val_loss:
                    best_val_loss = monitor
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 40:
                    if verbose:
                        print(f"      [{model_name}{model_id}] 早停 Epoch {epoch + 1}")
                    break

            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{epoch + 1},{avg_tl:.6f},{val_loss:.6f},{val_acc:.6f}\n")

            if verbose:
                acc_str = f"  Val Acc: {val_acc:.4f}" if val_acc >= 0 else ""
                vs = f"  Val Loss: {val_loss:.4f}{acc_str}" if val_loss >= 0 else ""
                print(f"      [{model_name}{model_id}] Epoch {epoch + 1:>3}/{epochs}"
                      f"  Train: {avg_tl:.4f}{vs}")

        if early_stop and best_state:
            model.load_state_dict(best_state)
        model.eval()
        return model, device

    def predict_rnn(model, device, history, seq_len=5, n_predict=5):
        """RNN 系列通用预测"""
        if model is None:
            return None
        numbers = np.array([rec['numbers'] for rec in history])
        current_seq = numbers[-seq_len:].copy()
        predictions = []
        with torch.no_grad():
            for _ in range(n_predict):
                x = torch.FloatTensor(current_seq).unsqueeze(0).to(device)
                outputs = model(x)
                pred, probs_l = [], []
                for p in range(POS_COUNT):
                    probs = torch.softmax(outputs[p], dim=-1).cpu().numpy()[0]
                    probs_l.append(probs)
                    pred.append(int(np.argmax(probs)))
                predictions.append({'numbers': pred, 'probs': probs_l})
                new_row = np.array(pred).reshape(1, POS_COUNT)
                current_seq = np.concatenate([current_seq[1:], new_row], axis=0)
        return predictions

    def method_rnn_ensemble(model_class, model_name, train_data, val_data,
                            full_history, n_predict=5, seq_len=5, n_models=3,
                            epochs=300, early_stop=True, log_dir='logs', run_tag=''):
        """多模型集成（LSTM/GRU通用）"""
        all_probs = [[np.zeros(POS_RANGES[p]) for p in range(POS_COUNT)]
                     for _ in range(n_predict)]

        os.makedirs(log_dir, exist_ok=True)

        for m_idx in range(n_models):
            print(f"\n    ┌─ {model_name} 模型 {m_idx + 1}/{n_models} ─┐")
            torch.manual_seed(42 + m_idx * 7)
            np.random.seed(42 + m_idx * 7)

            log_file = os.path.join(log_dir, f"{run_tag}_{model_name}_{m_idx + 1}.csv")
            model, device = train_rnn_model(
                model_class, train_data, val_data,
                seq_len=seq_len, epochs=epochs, lr=0.001,
                verbose=True, model_name=model_name, model_id=m_idx + 1,
                early_stop=early_stop, log_file=log_file
            )
            if model is None:
                continue

            preds = predict_rnn(model, device, full_history, seq_len, n_predict)
            if preds is None:
                continue

            for step in range(n_predict):
                for pos in range(POS_COUNT):
                    all_probs[step][pos] += preds[step]['probs'][pos]
            print(f"    └─ {model_name} 模型 {m_idx + 1} 完成 ─┘")

        predictions = []
        for step in range(n_predict):
            pred = []
            for pos in range(POS_COUNT):
                total = all_probs[step][pos]
                pred.append(int(np.argmax(total)) if total.sum() > 0
                            else np.random.randint(0, POS_RANGES[pos]))
            predictions.append(pred)
        return predictions, all_probs


# ############################################################
#  集成投票器（更新：12种算法）
# ############################################################

def ensemble_predict(history, train_data, val_data, n_predict=5, n_rounds=200, epochs=300, early_stop=True, n_candidates=3):
    """12种方法集成投票"""

    stat_methods = {
        '频率分析': (method_frequency, 1.0),
        '热号冷号': (method_hot_cold, 1.2),
        '马尔可夫链': (method_markov, 1.5),
        '加权近期': (method_weighted_recent, 1.3),
        '差值趋势': (method_delta_trend, 1.0),
        '模式匹配': (method_pattern_match, 1.4),
    }

    vote_matrix = [
        [np.zeros(POS_RANGES[p]) for p in range(POS_COUNT)]
        for _ in range(n_predict)
    ]

    all_method_preds = {}  # 保存各方法预测结果

    # ================================================================
    # ① 6种统计方法
    # ================================================================
    print(f"\n  ── ① 统计方法投票 ({len(stat_methods)} 种 × {n_rounds} 轮) ──")
    for name, (func, weight) in stat_methods.items():
        for _ in range(n_rounds):
            try:
                preds = func(history, n_predict)
                for step in range(n_predict):
                    for pos in range(POS_COUNT):
                        vote_matrix[step][pos][preds[step][pos]] += weight
            except Exception:
                continue
        print(f"    OK {name} 完成")

    # ================================================================
    # ② 贝叶斯概率模型
    # ================================================================
    print(f"\n  ── ② 贝叶斯概率模型 ──")
    try:
        bayesian_probs = method_bayesian_probs(history)
        bayesian_weight = 1.8
        bayesian_pred = []
        for pos in range(POS_COUNT):
            vote_matrix[0][pos] += bayesian_probs[pos] * bayesian_weight * n_rounds
            bayesian_pred.append(int(np.argmax(bayesian_probs[pos])))
        all_method_preds['贝叶斯'] = bayesian_pred

        # 后续期用采样
        for _ in range(n_rounds):
            try:
                preds = method_bayesian(history, n_predict)
                for step in range(1, n_predict):
                    for pos in range(POS_COUNT):
                        vote_matrix[step][pos][preds[step][pos]] += bayesian_weight
            except Exception:
                continue

        print(f"    OK 贝叶斯模型完成")
    except Exception as e:
        print(f"    ⚠ 贝叶斯模型失败: {e}")

    # ================================================================
    # ③ 随机森林
    # ================================================================
    if HAS_SKLEARN:
        print(f"\n  ── ③ 随机森林 ──")
        try:
            rf_probs = method_random_forest_probs(history)
            if rf_probs:
                rf_weight = 1.8
                rf_pred = []
                for pos in range(POS_COUNT):
                    vote_matrix[0][pos] += rf_probs[pos] * rf_weight * n_rounds
                    rf_pred.append(int(np.argmax(rf_probs[pos])))
                all_method_preds['随机森林'] = rf_pred

                # 后续期
                rf_preds_multi = method_random_forest(history, n_predict)
                for step in range(1, min(n_predict, len(rf_preds_multi))):
                    for pos in range(POS_COUNT):
                        vote_matrix[step][pos][rf_preds_multi[step][pos]] += rf_weight * n_rounds * 0.5

                print(f"    OK 随机森林完成")
            else:
                print(f"    ⚠ 随机森林: 数据不足")
        except Exception as e:
            print(f"    ⚠ 随机森林失败: {e}")
    else:
        print(f"\n  ⓘ scikit-learn 未安装，跳过随机森林 (pip install scikit-learn)")

    # ================================================================
    # ④ PyTorch LSTM
    # ================================================================
    lstm_preds = None
    lstm_probs = None
    if HAS_TORCH:
        print(f"\n  ── ④ LSTM (3模型集成) ──")
        t0 = time.time()
        try:
            seq_len = min(5, len(history) - 2)
            lstm_preds, lstm_probs = method_rnn_ensemble(
                LSTMModel, "LSTM", train_data, val_data, history,
                n_predict=n_predict, seq_len=seq_len, n_models=3,
                epochs=epochs, early_stop=early_stop,
                log_dir='logs', run_tag='rnn'
            )
            elapsed = time.time() - t0
            print(f"\n    OK LSTM 完成 ({elapsed:.1f}s)")
            all_method_preds['LSTM'] = lstm_preds[0] if lstm_preds else None

            lstm_weight = 2.0
            for step in range(n_predict):
                for pos in range(POS_COUNT):
                    vote_matrix[step][pos] += lstm_probs[step][pos] * lstm_weight * n_rounds
        except Exception as e:
            print(f"    ⚠ LSTM 失败: {e}")

    # ================================================================
    # ⑤ GRU
    # ================================================================
    gru_preds = None
    gru_probs = None
    if HAS_TORCH:
        print(f"\n  ── ⑤ GRU (3模型集成) ──")
        t0 = time.time()
        try:
            seq_len = min(5, len(history) - 2)
            gru_preds, gru_probs = method_rnn_ensemble(
                GRUModel, "GRU", train_data, val_data, history,
                n_predict=n_predict, seq_len=seq_len, n_models=3,
                epochs=epochs, early_stop=early_stop,
                log_dir='logs', run_tag='rnn'
            )
            elapsed = time.time() - t0
            print(f"\n    OK GRU 完成 ({elapsed:.1f}s)")
            all_method_preds['GRU'] = gru_preds[0] if gru_preds else None

            gru_weight = 2.0
            for step in range(n_predict):
                for pos in range(POS_COUNT):
                    vote_matrix[step][pos] += gru_probs[step][pos] * gru_weight * n_rounds
        except Exception as e:
            print(f"    ⚠ GRU 失败: {e}")

    # ================================================================
    # ⑥ GAN
    # ================================================================
    gan_preds_out = None
    if HAS_TORCH:
        print(f"\n  ── ⑥ GAN 生成对抗网络 ──")
        t0 = time.time()
        try:
            seq_len = min(5, len(history) - 2)
            torch.manual_seed(42)
            log_file = os.path.join('logs', f"rnn_gan_{int(time.time())}.csv")
            gen, g_device, noise_dim = train_gan(
                history, seq_len=seq_len, epochs=epochs, verbose=True,
                early_stop=early_stop, log_file=log_file
            )
            if gen is not None:
                gan_preds = predict_gan(gen, g_device, noise_dim, history,
                                        seq_len, n_predict)
                elapsed = time.time() - t0
                print(f"    OK GAN 完成 ({elapsed:.1f}s)")

                gan_weight = 1.5
                gan_preds_list = []
                for step in range(n_predict):
                    pred = gan_preds[step]['numbers']
                    gan_preds_list.append(pred)
                    for pos in range(POS_COUNT):
                        vote_matrix[step][pos] += gan_preds[step]['probs'][pos] * gan_weight * n_rounds

                gan_preds_out = gan_preds_list
                all_method_preds['GAN'] = gan_preds_list[0]
        except Exception as e:
            print(f"    ⚠ GAN 失败: {e}")

    # ================================================================
    # ⑦ 强化学习
    # ================================================================
    rl_preds_out = None
    if HAS_TORCH:
        print(f"\n  ── ⑦ 强化学习 Policy Gradient ──")
        t0 = time.time()
        try:
            torch.manual_seed(42)
            log_file = os.path.join('logs', f"rnn_rl_{int(time.time())}.csv")
            policy, rl_device, state_dim = train_rl_policy(
                history, lookback=10, epochs=epochs, verbose=True,
                early_stop=early_stop, log_file=log_file
            )
            if policy is not None:
                rl_preds = predict_rl(policy, rl_device, history, lookback=10,
                                      n_predict=n_predict)
                elapsed = time.time() - t0
                print(f"    OK 强化学习完成 ({elapsed:.1f}s)")

                rl_weight = 1.5
                rl_preds_list = []
                for step in range(n_predict):
                    pred = rl_preds[step]['numbers']
                    rl_preds_list.append(pred)
                    for pos in range(POS_COUNT):
                        vote_matrix[step][pos] += rl_preds[step]['probs'][pos] * rl_weight * n_rounds

                rl_preds_out = rl_preds_list
                all_method_preds['RL'] = rl_preds_list[0]
        except Exception as e:
            print(f"    ⚠ 强化学习失败: {e}")

    if not HAS_TORCH:
        print(f"\n  ⓘ PyTorch 未安装，跳过 LSTM/GRU/GAN/RL (pip install torch)")

    # ================================================================
    # 汇总
    # ================================================================
    final_predictions = []
    confidence_matrix = []

    for step in range(n_predict):
        candidates = []
        candidate_confs = []
        for _ in range(n_candidates):
            pred = []
            conf = []
            for pos in range(POS_COUNT):
                votes = vote_matrix[step][pos]
                total = votes.sum()
                if total == 0:
                    num = np.random.randint(0, POS_RANGES[pos])
                    prob = 1.0 / POS_RANGES[pos]
                else:
                    probs = votes / total
                    num = np.random.choice(POS_RANGES[pos], p=probs)
                    prob = probs[num]
                pred.append(num)
                conf.append(prob)
            candidates.append(pred)
            candidate_confs.append(conf)
        final_predictions.append(candidates)
        confidence_matrix.append(candidate_confs)

    return (final_predictions, confidence_matrix, vote_matrix,
            lstm_preds, gru_preds, gan_preds_out, rl_preds_out,
            all_method_preds)


# ############################################################
#  输出函数
# ############################################################

def fmt(num, pos):
    return f"{num:>2}" if pos == 6 else f"{num}"


def print_analysis(history):
    print(f"\n{'═' * 70}")
    print(f"  📊 历史数据统计分析")
    print(f"{'═' * 70}")

    n = len(history)
    print(f"\n  数据范围: {history[0]['issue']} ~ {history[-1]['issue']}")
    print(f"  总期数:   {n} 期")

    print(f"\n  {'─' * 66}")
    print(f"  位置1~6 频率 (0~9)")
    print(f"  {'─' * 66}")
    print(f"  {'位置':>6}", end="")
    for d in range(10):
        print(f"  {d:>4}", end="")
    print(f"   最热")
    print(f"  {'─' * 66}")

    for pos in range(6):
        nums = [r['numbers'][pos] for r in history]
        c = Counter(nums)
        print(f"  位置{pos + 1}", end="")
        for d in range(10):
            print(f"  {c.get(d, 0):>4}", end="")
        mc = c.most_common(1)[0]
        print(f"   {mc[0]}({mc[1]})")

    print(f"\n  {'─' * 78}")
    print(f"  位置7 频率 (0~14)")
    print(f"  {'─' * 78}")
    c7 = Counter(r['numbers'][6] for r in history)
    print(f"  {'':>6}", end="")
    for d in range(15):
        print(f"  {d:>4}", end="")
    print(f"   最热")
    print(f"  {'─' * 78}")
    print(f"  位置7", end="")
    for d in range(15):
        print(f"  {c7.get(d, 0):>4}", end="")
    mc7 = c7.most_common(1)[0]
    print(f"   {mc7[0]}({mc7[1]})")

    print(f"\n  {'─' * 66}")
    print(f"  近5期热号/冷号")
    print(f"  {'─' * 66}")
    recent5 = history[-5:]
    for pos in range(POS_COUNT):
        n_vals = POS_RANGES[pos]
        rn = [r['numbers'][pos] for r in recent5]
        c = Counter(rn)
        hot = [str(k) for k, v in c.most_common() if v >= 2]
        cold = [str(d) for d in range(n_vals) if d not in set(rn)]
        print(f"  位置{pos + 1}(0~{n_vals - 1:>2}): {rn}  "
              f"热[{','.join(hot) or '无'}]  冷[{','.join(cold[:5])}]")

    print(f"\n  {'─' * 66}")
    print(f"  最近10期奇偶比")
    print(f"  {'─' * 66}")
    for rec in history[-10:]:
        nums = rec['numbers']
        odd = sum(1 for x in nums if x % 2 == 1)
        bar = "█" * odd + "░" * (7 - odd)
        ns = ' '.join(fmt(nums[p], p) for p in range(POS_COUNT))
        print(f"  {rec['issue']}  {ns}  奇{odd}:偶{7 - odd}  {bar}")

    sums = [sum(r['numbers']) for r in history]
    rs = sums[-10:]
    print(f"\n  和值: 全局均值={np.mean(sums):.1f}  近10期={np.mean(rs):.1f}  "
          f"范围=[{min(rs)},{max(rs)}]")


def print_position_probs(vote_matrix, step, pred):
    print(f"\n  位置1~6:")
    print(f"  {'位置':>6}", end="")
    for d in range(10):
        print(f" {d:>5}", end="")
    print(f"  选中")
    print(f"  {'─' * 66}")
    for pos in range(6):
        votes = vote_matrix[step][pos]
        total = votes.sum()
        probs = votes / total if total > 0 else np.zeros(POS_RANGES[pos])
        print(f"  位置{pos + 1}", end="")
        for d in range(10):
            pct = probs[d] * 100
            if d == pred[pos]:
                print(f" {pct:>4.0f}▓", end="")
            elif pct >= 10:
                print(f" {pct:>4.0f}░", end="")
            else:
                print(f" {pct:>4.0f} ", end="")
        print(f"  → {pred[pos]}")

    print(f"\n  位置7 (0~14):")
    print(f"  {'':>6}", end="")
    for d in range(15):
        print(f" {d:>5}", end="")
    print(f"  选中")
    print(f"  {'─' * 84}")
    v7 = vote_matrix[step][6]
    t7 = v7.sum()
    p7 = v7 / t7 if t7 > 0 else np.zeros(15)
    print(f"  位置7", end="")
    for d in range(15):
        pct = p7[d] * 100
        if d == pred[6]:
            print(f" {pct:>4.0f}▓", end="")
        elif pct >= 10:
            print(f" {pct:>4.0f}░", end="")
        else:
            print(f" {pct:>4.0f} ", end="")
    print(f"  → {pred[6]}")


# ############################################################
#  主程序
# ############################################################

def main():
    n_algo = 6 + (4 if HAS_TORCH else 0) + 1 + (1 if HAS_SKLEARN else 0)

    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║         7 星 彩 号 码 智 能 预 测 系 统  v4.0                    ║")
    print("║       12-Algorithm Ensemble: Stat + DL + GAN + RL + Bayesian     ║")
    print("║       ★ 位置7范围0~14 | 训练/验证/测试划分 | 回测评估           ║")
    print("╠════════════════════════════════════════════════════════════════════╣")
    print(f"║  号码: 位1~6 [0-9]  位7 [0-14]                                   ║")
    print(f"║  PyTorch:     {'OK' if HAS_TORCH else 'NO pip install torch':<48} ║")
    print(f"║  sklearn:     {'OK' if HAS_SKLEARN else 'NO pip install scikit-learn':<48} ║")
    print(f"║  可用算法:    {n_algo} 种                                            ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    # 用户输入参数
    try:
        epochs = int(input("请输入模型训练轮数 (默认500): ") or 500)
        early_stop = input("是否启用早停 (y/n, 默认n): ").lower().strip()
        early_stop = early_stop != 'n'
    except ValueError:
        epochs = 500
        early_stop = False

    print(f"训练轮数: {epochs}, 早停: {'启用' if early_stop else '禁用'}")

    # ---- 加载数据 ----
    try:
        history = load_data('7星彩_历史开奖号码.csv', 'raw_api_response.json')
        print(f"  OK 从CSV和JSON加载 {len(history)} 条历史记录")
    except FileNotFoundError as e:
        if '7星彩_历史开奖号码.csv' in str(e):
            print(f"\n  未找到CSV，使用内置数据...")
            history = get_builtin_data()
            print(f"  OK 加载 {len(history)} 条内置记录")
        else:
            print(f"\n  未找到JSON，使用CSV数据...")
            try:
                history = load_data_from_csv('7星彩_历史开奖号码.csv')
                print(f"  OK 从CSV加载 {len(history)} 条历史记录")
            except FileNotFoundError:
                print(f"\n  未找到CSV，使用内置数据...")
                history = get_builtin_data()
                print(f"  OK 加载 {len(history)} 条内置记录")

    p7_vals = [r['numbers'][6] for r in history]
    p7_two = sum(1 for v in p7_vals if v >= 10)
    print(f"  位置7: max={max(p7_vals)}, 两位数={p7_two}条 ({p7_two / len(history) * 100:.1f}%)")

    # ---- 数据集划分 ----
    train_data, val_data, test_data = split_dataset(history, 0.70, 0.15)

    print(f"\n  {'═' * 66}")
    print(f"  📂 数据集划分")
    print(f"  {'═' * 66}")
    print(f"  训练集:  {len(train_data):>4} 期  ({train_data[0]['issue']}~{train_data[-1]['issue']})")
    if val_data:
        print(f"  验证集:  {len(val_data):>4} 期  ({val_data[0]['issue']}~{val_data[-1]['issue']})")
    if test_data:
        print(f"  测试集:  {len(test_data):>4} 期  ({test_data[0]['issue']}~{test_data[-1]['issue']})")
    print(f"  总计:    {len(history):>4} 期")

    # ---- 分析 ----
    print_analysis(history)

    # ============================================================
    # 正式预测
    # ============================================================
    n_predict = 5

    print(f"\n{'═' * 70}")
    print(f"  🎯 集成预测启动 ({n_algo} 种算法)")
    print(f"{'═' * 70}")
    print(f"  ① 统计方法 x6  ② 贝叶斯  ③ 随机森林  ④ LSTM  ⑤ GRU  ⑥ GAN  ⑦ RL")

    t_start = time.time()
    (predictions, confidences, vote_matrix,
     lstm_preds, gru_preds, gan_preds, rl_preds,
     all_method_preds) = ensemble_predict(
        history, train_data, val_data, n_predict=n_predict, epochs=epochs, early_stop=early_stop, n_candidates=3
    )
    t_total = time.time() - t_start

    last_issue = int(history[-1]['issue'])

    # 保存完整日志
    log_file_path = save_comprehensive_log(
        history, predictions, confidences, all_method_preds,
        lstm_preds, gru_preds, gan_preds, rl_preds,
        vote_matrix, t_total, last_issue, n_predict
    )
    print(f"\n  ✅ 完整预测日志已保存: {log_file_path}")

    # ============================================================
    # 结果展示
    # ============================================================
    print(f"\n{'═' * 70}")
    print(f"  🏆 最终预测结果 (耗时 {t_total:.1f}s)")
    print(f"{'═' * 70}")

    # 表头
    print(f"\n  {'期号':>8}  {'候选号码':>18}  {'中奖概率':>8}", end="")
    has_extra = []
    for name, arr in [('LSTM', lstm_preds), ('GRU', gru_preds),
                       ('GAN', gan_preds), ('RL', rl_preds)]:
        if arr:
            has_extra.append((name, arr))
            print(f"  {name:>12}", end="")
    print()
    print(f"  {'─' * (50 + 14 * len(has_extra))}")

    for step in range(n_predict):
        issue = last_issue + step + 1
        candidates = predictions[step]
        candidate_confs = confidences[step]
        print(f"\n  第 {issue} 期:")
        for i, (pred, conf) in enumerate(zip(candidates, candidate_confs)):
            prob = np.prod(conf)  # 联合概率近似
            ps = ' '.join(fmt(pred[p], p) for p in range(POS_COUNT))
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"    候选{i+1}: [{ps}]  {prob * 100:>6.1f}%  {bar}")
            if has_extra:
                print("    ", end="")
                for name, arr in has_extra:
                    ep = arr[step] if step < len(arr) else None
                    if ep:
                        es = ' '.join(fmt(ep[p], p) for p in range(POS_COUNT))
                        print(f"  [{es}]", end="")
                    else:
                        print(f"  {'--':>12}", end="")
                print()

    # ---- 详细概率 ----
    print(f"\n{'═' * 70}")
    print(f"  📋 第1期详细概率分布")
    print(f"{'═' * 70}")
    print_position_probs(vote_matrix, 0, predictions[0][0])

    # ---- 备选号码 ----
    print(f"\n{'═' * 70}")
    print(f"  🔄 备选号码 (Top3)")
    print(f"{'═' * 70}")
    for step in range(n_predict):
        issue = last_issue + step + 1
        print(f"\n  第 {issue} 期:")
        for pos in range(POS_COUNT):
            votes = vote_matrix[step][pos]
            total = votes.sum()
            probs = votes / total if total > 0 else np.zeros(POS_RANGES[pos])
            top3 = np.argsort(probs)[::-1][:3]
            t3s = "  ".join(f"{idx}({probs[idx] * 100:.0f}%)" for idx in top3)
            print(f"    位置{pos + 1}(0~{POS_RANGES[pos] - 1:>2}): {t3s}")

    # ---- 各算法独立预测 ----
    print(f"\n{'═' * 70}")
    print(f"  🔬 各算法独立预测（第1期）")
    print(f"{'═' * 70}")

    np.random.seed(42)
    for name, func in [
        ("频率分析", method_frequency), ("热号冷号", method_hot_cold),
        ("马尔可夫链", method_markov), ("加权近期", method_weighted_recent),
        ("差值趋势", method_delta_trend), ("模式匹配", method_pattern_match),
    ]:
        p = func(history, 1)
        ps = ' '.join(fmt(p[0][i], i) for i in range(POS_COUNT))
        print(f"  {name:>10}: [{ps}]  和值={sum(p[0])}")

    for name in ['贝叶斯', '随机森林', 'LSTM', 'GRU', 'GAN', 'RL']:
        if name in all_method_preds and all_method_preds[name]:
            p = all_method_preds[name]
            ps = ' '.join(fmt(p[i], i) for i in range(POS_COUNT))
            print(f"  {name:>10}: [{ps}]  和值={sum(p)}")

    # ---- 一致性 ----
    print(f"\n{'═' * 70}")
    print(f"  📊 算法一致性分析（第1期）")
    print(f"{'═' * 70}")
    for pos in range(POS_COUNT):
        nv = POS_RANGES[pos]
        votes = vote_matrix[0][pos]
        total = votes.sum()
        probs = votes / total if total > 0 else np.zeros(nv)
        top1 = int(np.argmax(probs))
        top1_pct = probs[top1] * 100
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
        consensus = max(0, 1 - entropy / np.log2(nv))
        bar = "█" * int(consensus * 20) + "░" * (20 - int(consensus * 20))
        label = '✅高' if consensus > 0.4 else '⚠️低' if consensus < 0.2 else '→中'
        print(f"  位置{pos + 1}(0~{nv - 1:>2}): 首选={top1}({top1_pct:.0f}%)  "
              f"一致性={consensus:.2f} {bar} {label}")

    # ---- 最终推荐 ----
    print(f"\n{'═' * 70}")
    print(f"  💡 最终推荐号码")
    print(f"{'═' * 70}")
    for step in range(n_predict):
        issue = last_issue + step + 1
        pred = predictions[step][0]  # 使用第一个候选
        conf = np.mean(confidences[step])
        stars = "⭐" * min(5, max(1, int(conf * 10)))
        ps = '  '.join(fmt(pred[p], p) for p in range(POS_COUNT))
        print(f"\n  📌 第 {issue} 期:  [ {ps} ]  {stars}  置信度 {conf * 100:.1f}%")

    print(f"\n{'═' * 70}")
    print(f"  ⚠️  声明: 彩票开奖完全随机，任何算法（包括深度学习）")
    print(f"  ⚠️  都无法真正预测随机事件。以上仅供学习和娱乐参考。")
    print(f"  ⚠️  请理性购彩，量力而行。")
    print(f"{'═' * 70}\n")


def save_comprehensive_log(history, predictions, confidences, all_method_preds, 
                          lstm_preds, gru_preds, gan_preds, rl_preds,
                          vote_matrix, t_total, last_issue, n_predict=5):
    """保存完整的预测结果到日志文件"""
    import json
    from datetime import datetime
    
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 生成时间戳文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f"comprehensive_prediction_{timestamp}.log")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        # 写入头部信息
        f.write("=" * 80 + "\n")
        f.write(f"7星彩智能预测系统 - 完整日志\n")
        f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"耗时: {t_total:.1f}秒\n")
        f.write(f"历史数据期数: {len(history)}\n")
        f.write(f"预测期数: {n_predict}\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入最终预测结果
        f.write("【最终集成预测结果】\n")
        f.write("-" * 50 + "\n")
        for step in range(n_predict):
            issue = last_issue + step + 1
            candidates = predictions[step]
            candidate_confs = confidences[step]
            f.write(f"\n第 {issue} 期预测:\n")
            for i, (pred, conf) in enumerate(zip(candidates, candidate_confs)):
                prob = np.prod(conf)
                pred_str = ' '.join(str(pred[p]) for p in range(POS_COUNT))
                f.write(f"  候选{i+1}: [{pred_str}] 置信度: {prob*100:.2f}%\n")
        
        # 写入各算法独立预测结果
        f.write("\n\n【各算法独立预测结果（第1期）】\n")
        f.write("-" * 50 + "\n")
        
        # 统计方法
        stat_methods = [
            ("频率分析", method_frequency), ("热号冷号", method_hot_cold),
            ("马尔可夫链", method_markov), ("加权近期", method_weighted_recent),
            ("差值趋势", method_delta_trend), ("模式匹配", method_pattern_match),
        ]
        for name, func in stat_methods:
            p = func(history, 1)
            ps = ' '.join(str(p[0][i]) for i in range(POS_COUNT))
            f.write(f"{name:>10}: [{ps}] 和值={sum(p[0])}\n")
        
        # 机器学习/深度学习方法
        ml_methods = ['贝叶斯', '随机森林', 'LSTM', 'GRU', 'GAN', 'RL']
        for name in ml_methods:
            if name in all_method_preds and all_method_preds[name]:
                p = all_method_preds[name]
                ps = ' '.join(str(p[i]) for i in range(POS_COUNT))
                f.write(f"{name:>10}: [{ps}] 和值={sum(p)}\n")
        
        # 写入详细概率分布
        f.write("\n\n【第1期详细概率分布】\n")
        f.write("-" * 50 + "\n")
        for pos in range(POS_COUNT):
            votes = vote_matrix[0][pos]
            total = votes.sum()
            probs = votes / total if total > 0 else np.zeros(POS_RANGES[pos])
            top5 = np.argsort(probs)[::-1][:5]
            f.write(f"位置{pos + 1}(0~{POS_RANGES[pos] - 1}): ")
            for idx in top5:
                f.write(f"{idx}({probs[idx]*100:.1f}%) ")
            f.write("\n")
        
        # 写入备选号码
        f.write("\n\n【备选号码 Top3（各期）】\n")
        f.write("-" * 50 + "\n")
        for step in range(n_predict):
            issue = last_issue + step + 1
            f.write(f"\n第 {issue} 期:\n")
            for pos in range(POS_COUNT):
                votes = vote_matrix[step][pos]
                total = votes.sum()
                probs = votes / total if total > 0 else np.zeros(POS_RANGES[pos])
                top3 = np.argsort(probs)[::-1][:3]
                t3s = " ".join(f"{idx}({probs[idx]*100:.0f}%)" for idx in top3)
                f.write(f"  位置{pos + 1}: {t3s}\n")
        
        # 写入算法一致性分析
        f.write("\n\n【算法一致性分析（第1期）】\n")
        f.write("-" * 50 + "\n")
        for pos in range(POS_COUNT):
            nv = POS_RANGES[pos]
            votes = vote_matrix[0][pos]
            total = votes.sum()
            probs = votes / total if total > 0 else np.zeros(nv)
            top1 = int(np.argmax(probs))
            top1_pct = probs[top1] * 100
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
            consensus = max(0, 1 - entropy / np.log2(nv))
            f.write(f"位置{pos + 1}: 首选={top1}({top1_pct:.0f}%) 一致性={consensus:.3f}\n")
        
        # 写入历史数据统计
        f.write("\n\n【历史数据统计】\n")
        f.write("-" * 50 + "\n")
        sums = [sum(r['numbers']) for r in history]
        f.write(f"总期数: {len(history)}\n")
        f.write(f"和值均值: {np.mean(sums):.2f}\n")
        f.write(f"和值范围: [{min(sums)}, {max(sums)}]\n")
        p7_vals = [r['numbers'][6] for r in history]
        p7_two = sum(1 for v in p7_vals if v >= 10)
        f.write(f"位置7最大值: {max(p7_vals)}\n")
        f.write(f"位置7两位数比例: {p7_two / len(history) * 100:.1f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("日志结束\n")
        f.write("=" * 80 + "\n")
    
    return log_file


if __name__ == '__main__':
    main()
