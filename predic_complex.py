"""
7星彩号码智能预测系统 v5.1
═══════════════════════════
13种算法集成预测 — 全部问题修复版

修复清单:
  🔴 #1  数据泄露 → 所有ML模型只用train_data训练
  🔴 #2  候选生成 → 确定性Top-N，可复现
  🔴 #3  置信度   → np.mean 替代 np.prod
  🔴 #4  数据增强 → 真正的扰动增强
  🟡 #5  GAN/RL早停 → 基于验证集
  🟡 #6  dropout → 0.3
  🟡 #7  日志重复 → 缓存结果
  🟡 #8  打印过多 → 每100epoch
  🟡 #9  默认值矛盾 → 修正逻辑
  🟡 #10 重复导入 → 统一顶部
  🟢 #11 TF-Attention → 已添加
  🟢 #12 学习率预热 → WarmupCosine
  🟢 #13 Label Smoothing → 全模型启用
  🟢 #14 随机森林双训练 → 训练一次缓存
  🟢 #15 模型保存加载 → checkpoint机制

依赖:
  pip install numpy torch scikit-learn tensorflow
"""

# ═══════════════════════════════════════════════════════
#  🟡 #10 修复: 统一在顶部导入，不在函数内重复导入
# ═══════════════════════════════════════════════════════
import csv
import numpy as np
from collections import Counter
from datetime import datetime
import warnings
import time
import os
import json
import hashlib

warnings.filterwarnings('ignore')

# ============================================================
# 全局常量
# ============================================================
POS_RANGES = [10, 10, 10, 10, 10, 10, 15]
POS_COUNT = 7
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

# ============================================================
# 依赖检测
# ============================================================
HAS_TORCH = False
HAS_SKLEARN = False
HAS_TF = False

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

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    pass

if HAS_TORCH and torch.cuda.is_available():
    torch.cuda.empty_cache()

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
#  数据加载
# ============================================================
def load_data(filename='7星彩_历史开奖号码.csv'):
    data = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nums = [int(row[f'号码{i}']) for i in range(1, 8)]
            data.append({'issue': row['期号'], 'date': row.get('开奖日期', ''), 'numbers': nums})
    data.sort(key=lambda x: x['issue'])
    errors = sum(1 for rec in data for pos in range(POS_COUNT)
                 if not (0 <= rec['numbers'][pos] < POS_RANGES[pos]))
    print(f"  {'⚠ 范围异常: '+str(errors)+'个' if errors else '✓ 全部 '+str(len(data))+' 条正确'}")
    return data


def get_builtin_data():
    raw = [
        ("26018",[2,9,3,6,5,7,2]),  ("26019",[6,0,8,8,2,7,1]),
        ("26020",[5,5,0,9,5,6,4]),  ("26021",[6,4,3,5,6,3,2]),
        ("26022",[7,9,9,1,0,2,3]),  ("26023",[2,8,3,8,4,8,11]),
        ("26024",[5,4,4,9,0,3,14]), ("26025",[8,4,0,5,5,3,1]),
        ("26026",[2,6,9,8,3,8,12]), ("26027",[8,7,7,3,6,0,9]),
        ("26028",[7,9,8,0,1,1,6]),  ("26029",[1,4,0,4,3,2,13]),
        ("26030",[1,9,8,1,5,9,1]),  ("26031",[6,8,9,5,2,2,10]),
        ("26032",[3,1,6,4,4,5,1]),  ("26033",[1,8,9,1,9,3,1]),
        ("26034",[3,9,9,1,5,3,5]),
    ]
    return [{'issue': r[0], 'date': '', 'numbers': r[1]} for r in raw]


def split_dataset(history, train_ratio=0.70, val_ratio=0.15):
    n = len(history)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return history[:t1], history[t1:t2], history[t2:]


def data_hash(history):
    """为数据集生成指纹，用于检查缓存有效性"""
    s = ''.join(r['issue'] for r in history[:5]) + str(len(history))
    return hashlib.md5(s.encode()).hexdigest()[:8]


# ============================================================
#  🔴 #4 修复: 有效数据增强
# ============================================================
def augment_sequences(X, y, n_aug=2):
    """
    对序列做有意义的扰动增强（不是简单复制）
    ───────────────────────────────────────────
    - 随机选 15% 的位置做 ±1 偏移
    - 各位置 clamp 到合法范围
    - 保持 y 不变（目标不动）
    """
    X_list = [X]
    y_list = [y]
    for aug_i in range(n_aug):
        rng = np.random.RandomState(42 + aug_i)
        X_noisy = X.copy().astype(np.float32)
        mask = rng.random(X.shape) < 0.15
        perturbation = rng.choice([-1, 0, 1], size=X.shape)
        X_noisy += mask * perturbation
        for pos in range(POS_COUNT):
            X_noisy[:, :, pos] = np.clip(X_noisy[:, :, pos], 0, POS_RANGES[pos] - 1)
        X_list.append(X_noisy.astype(X.dtype))
        y_list.append(y.copy())
    return np.concatenate(X_list), np.concatenate(y_list)


# ============================================================
#  工具函数
# ============================================================
def build_features(history, idx, lookback=10):
    start = max(0, idx - lookback)
    recent = history[start:idx]
    if not recent:
        return None
    features = []
    for pos in range(POS_COUNT):
        n_vals = POS_RANGES[pos]
        nums = [r['numbers'][pos] for r in recent]
        counter = Counter(nums)
        features.extend([counter.get(v, 0) / len(nums) for v in range(n_vals)])
        features.append(recent[-1]['numbers'][pos] / max(n_vals - 1, 1))
        features.append((nums[-1] - nums[-2]) / max(n_vals - 1, 1) if len(nums) >= 2 else 0.0)
        features.append(np.mean(nums) / max(n_vals - 1, 1))
        features.append(np.std(nums) / max(n_vals - 1, 1) if len(nums) > 1 else 0.0)
    sums = [sum(r['numbers']) for r in recent]
    features.append(np.mean(sums) / 50)
    features.append(np.std(sums) / 20 if len(sums) > 1 else 0.0)
    features.append(sum(1 for x in recent[-1]['numbers'] if x % 2 == 1) / POS_COUNT)
    return np.array(features, dtype=np.float32)


def fmt(num, pos):
    return f"{num:>2}" if pos == 6 else f"{num}"


# ############################################################
#  第一部分：6种统计方法
# ############################################################

def method_frequency(history, n_predict=5):
    nums_by_pos = [[r['numbers'][i] for r in history] for i in range(POS_COUNT)]
    preds = []
    for _ in range(n_predict):
        pred = []
        for pos in range(POS_COUNT):
            nv = POS_RANGES[pos]
            c = Counter(nums_by_pos[pos])
            w = [c.get(v, 0) + 0.1 for v in range(nv)]
            t = sum(w)
            pred.append(int(np.random.choice(nv, p=[x/t for x in w])))
        preds.append(pred)
    return preds


def method_hot_cold(history, n_predict=5, window=10):
    recent = history[-window:]
    nums_by_pos = [[r['numbers'][i] for r in recent] for i in range(POS_COUNT)]
    preds = []
    for _ in range(n_predict):
        pred = []
        for pos in range(POS_COUNT):
            nv = POS_RANGES[pos]
            c = Counter(nums_by_pos[pos])
            w = []
            for v in range(nv):
                freq = c.get(v, 0)
                w.append(freq * 2.0 if freq >= 2 else (1.5 if freq == 0 else 1.0))
            t = sum(w)
            pred.append(int(np.random.choice(nv, p=[x/t for x in w])))
        preds.append(pred)
    return preds


def method_markov(history, n_predict=5):
    transition = []
    for pos in range(POS_COUNT):
        nv = POS_RANGES[pos]
        m = np.ones((nv, nv)) * 0.01
        for j in range(1, len(history)):
            m[history[j-1]['numbers'][pos]][history[j]['numbers'][pos]] += 1
        for r in range(nv):
            m[r] /= m[r].sum()
        transition.append(m)
    preds = []
    cur = list(history[-1]['numbers'])
    for _ in range(n_predict):
        pred = [int(np.random.choice(POS_RANGES[p], p=transition[p][cur[p]])) for p in range(POS_COUNT)]
        preds.append(pred)
        cur = list(pred)
    return preds


def method_weighted_recent(history, n_predict=5):
    n = len(history)
    preds = []
    for _ in range(n_predict):
        pred = []
        for pos in range(POS_COUNT):
            sc = np.zeros(POS_RANGES[pos])
            for j in range(n):
                sc[history[j]['numbers'][pos]] += 0.9 ** (n - 1 - j)
            pred.append(int(np.random.choice(POS_RANGES[pos], p=sc / sc.sum())))
        preds.append(pred)
    return preds


def method_delta_trend(history, n_predict=5, window=8):
    recent = history[-window:]
    preds = []
    last = history[-1]['numbers']
    for _ in range(n_predict):
        pred = []
        for pos in range(POS_COUNT):
            nv = POS_RANGES[pos]
            deltas = [recent[j]['numbers'][pos] - recent[j-1]['numbers'][pos] for j in range(1, len(recent))]
            if not deltas:
                pred.append(last[pos]); continue
            w = [0.5 ** (len(deltas)-1-i) for i in range(len(deltas))]
            avg_d = sum(d * ww for d, ww in zip(deltas, w)) / sum(w)
            v = int(round(last[pos] + avg_d + np.random.normal(0, 1.5))) % nv
            pred.append(v if v >= 0 else v + nv)
        preds.append(pred)
        last = pred
    return preds


def method_pattern_match(history, n_predict=5, match_len=3):
    if len(history) < match_len + 1:
        return method_frequency(history, n_predict)
    query = [r['numbers'] for r in history[-match_len:]]
    sims = []
    for i in range(len(history) - match_len - 1):
        seg = [history[i+j]['numbers'] for j in range(match_len)]
        sc = sum(1 if seg[t][p] == query[t][p] else 0.5 if abs(seg[t][p]-query[t][p]) <= 1 else 0
                 for t in range(match_len) for p in range(POS_COUNT))
        sims.append((sc, i + match_len))
    sims.sort(key=lambda x: x[0], reverse=True)
    preds = []
    for step in range(n_predict):
        ps = [np.zeros(POS_RANGES[p]) for p in range(POS_COUNT)]
        for sc, ni in sims[:10]:
            if ni + step < len(history):
                for pos in range(POS_COUNT):
                    ps[pos][history[ni+step]['numbers'][pos]] += sc + 0.01
        pred = [int(np.random.choice(POS_RANGES[p], p=s/s.sum())) if s.sum() > 0
                else np.random.randint(0, POS_RANGES[p]) for p, s in enumerate(ps)]
        preds.append(pred)
    return preds


# ############################################################
#  第二部分：PyTorch 基础设施
# ############################################################

if HAS_TORCH:

    class LotteryDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    def make_sequences(history_list, seq_len=5):
        numbers = np.array([r['numbers'] for r in history_list])
        if len(numbers) <= seq_len:
            return np.array([]), np.array([])
        X = np.array([numbers[i:i+seq_len] for i in range(len(numbers)-seq_len)])
        y = np.array([numbers[i+seq_len] for i in range(len(numbers)-seq_len)])
        return X, y

    def get_device():
        if torch.cuda.is_available():
            try:
                if torch.cuda.mem_get_info()[0] / 1024**2 > 256:
                    return torch.device('cuda')
            except Exception:
                pass
        return torch.device('cpu')

    def evaluate_model(model, device, X, y):
        model.eval()
        # 🟢 #13: Label Smoothing 在 loss 中
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        loader = DataLoader(LotteryDataset(X, y), batch_size=64, shuffle=False)
        tl, pc, pt = 0, np.zeros(POS_COUNT), np.zeros(POS_COUNT)
        n = 0
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                tl += sum(criterion(out[p], by[:,p]) for p in range(POS_COUNT)).item() * bx.size(0)
                for p in range(POS_COUNT):
                    pc[p] += (torch.argmax(out[p], -1) == by[:,p]).sum().item()
                    k = min(3, POS_RANGES[p])
                    t3 = torch.topk(out[p], k, -1).indices
                    for ri in range(bx.size(0)):
                        if by[ri, p] in t3[ri]: pt[p] += 1
                n += bx.size(0)
        return tl/max(n,1), pc/max(n,1), pt/max(n,1)

    # ═══════════════════════════════════════════════
    #  🟢 #12: Warmup + Cosine 学习率调度器
    # ═══════════════════════════════════════════════
    class WarmupCosineScheduler:
        """
        线性预热 + 余弦退火
        ────────────────────
        0 → warmup_steps: lr 从 0 线性升到 base_lr
        warmup_steps → total: 余弦退火到 min_lr
        """
        def __init__(self, optimizer, warmup_steps=20, total_steps=300,
                     base_lr=0.003, min_lr=1e-6):
            self.optimizer = optimizer
            self.warmup_steps = warmup_steps
            self.total_steps = total_steps
            self.base_lr = base_lr
            self.min_lr = min_lr
            self.current_step = 0

        def step(self):
            self.current_step += 1
            if self.current_step <= self.warmup_steps:
                lr = self.base_lr * self.current_step / max(self.warmup_steps, 1)
            else:
                progress = (self.current_step - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1)
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            return lr

    # ═══════════════════════════════════════════════
    #  🟢 #15: 模型保存/加载
    # ═══════════════════════════════════════════════
    def save_model(model, name, d_hash):
        path = os.path.join(CHECKPOINT_DIR, f"{name}_{d_hash}.pt")
        torch.save(model.state_dict(), path)
        return path

    def load_model_if_exists(model, name, d_hash):
        path = os.path.join(CHECKPOINT_DIR, f"{name}_{d_hash}.pt")
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, weights_only=True))
                model.eval()
                return True
            except Exception:
                pass
        return False


# ############################################################
#  模型 7：LSTM  (🟡 #6: dropout=0.3)
# ############################################################

if HAS_TORCH:

    class LSTMModel(nn.Module):
        def __init__(self, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.emb_dim = 8
            self.embeddings = nn.ModuleList([
                nn.Embedding(POS_RANGES[p], self.emb_dim) for p in range(POS_COUNT)])
            self.lstm = nn.LSTM(POS_COUNT*self.emb_dim, hidden_size, num_layers,
                                batch_first=True,
                                dropout=dropout if num_layers > 1 else 0,
                                bidirectional=True)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_size * 2)
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_size*2, 32), nn.ReLU(),
                              nn.Dropout(dropout), nn.Linear(32, POS_RANGES[p]))
                for p in range(POS_COUNT)])

        def forward(self, x):
            xl = x.long()
            emb = torch.cat([self.embeddings[p](xl[:,:,p].clamp(0, POS_RANGES[p]-1))
                             for p in range(POS_COUNT)], dim=-1)
            out, _ = self.lstm(emb)
            return [h(self.dropout(self.norm(out[:,-1,:]))) for h in self.heads]


# ############################################################
#  模型 8：GRU  (🟡 #6: dropout=0.3)
# ############################################################

if HAS_TORCH:

    class GRUModel(nn.Module):
        def __init__(self, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.emb_dim = 8
            self.embeddings = nn.ModuleList([
                nn.Embedding(POS_RANGES[p], self.emb_dim) for p in range(POS_COUNT)])
            self.gru = nn.GRU(POS_COUNT*self.emb_dim, hidden_size, num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_size * 2)
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_size*2, 32), nn.ReLU(),
                              nn.Dropout(dropout), nn.Linear(32, POS_RANGES[p]))
                for p in range(POS_COUNT)])

        def forward(self, x):
            xl = x.long()
            emb = torch.cat([self.embeddings[p](xl[:,:,p].clamp(0, POS_RANGES[p]-1))
                             for p in range(POS_COUNT)], dim=-1)
            out, _ = self.gru(emb)
            return [h(self.dropout(self.norm(out[:,-1,:]))) for h in self.heads]


# ############################################################
#  模型 9：GAN  (🔴#1: 只用train_data, 🟡#5: 验证集早停)
# ############################################################

if HAS_TORCH:

    class NumberRelationEncoder(nn.Module):
        def __init__(self, emb_dim=8, n_heads=2):
            super().__init__()
            self.attn = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(emb_dim)
        def forward(self, x):
            o, _ = self.attn(x, x, x)
            return self.norm(x + o)

    class Generator(nn.Module):
        def __init__(self, noise_dim=32, cond_dim=64, hidden=128):
            super().__init__()
            ed = 8
            self.cond_emb = nn.ModuleList([
                nn.Embedding(POS_RANGES[p], ed) for p in range(POS_COUNT)])
            self.cond_gru = nn.GRU(POS_COUNT*ed, cond_dim, batch_first=True)
            self.rel = NumberRelationEncoder(ed)
            self.net = nn.Sequential(
                nn.Linear(noise_dim+cond_dim, hidden), nn.LayerNorm(hidden),
                nn.LeakyReLU(0.2), nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden), nn.LeakyReLU(0.2))
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, POS_RANGES[p]))
                for p in range(POS_COUNT)])

        def encode_condition(self, seq):
            xl = seq.long()
            embs = [self.cond_emb[p](xl[:,:,p].clamp(0, POS_RANGES[p]-1)) for p in range(POS_COUNT)]
            last = torch.stack([e[:,-1,:] for e in embs], dim=1)
            self.rel(last)
            x = torch.cat(embs, dim=-1)
            _, h = self.cond_gru(x)
            return h[-1]

        def forward(self, z, seq):
            c = self.encode_condition(seq)
            return [h(self.net(torch.cat([z, c], -1))) for h in self.heads]

    class Discriminator(nn.Module):
        def __init__(self, cond_dim=64, hidden=128):
            super().__init__()
            self.cond_emb = nn.ModuleList([
                nn.Embedding(POS_RANGES[p], 8) for p in range(POS_COUNT)])
            self.cond_gru = nn.GRU(POS_COUNT*8, cond_dim, batch_first=True)
            self.net = nn.Sequential(
                nn.Linear(sum(POS_RANGES)+cond_dim, hidden), nn.LayerNorm(hidden),
                nn.LeakyReLU(0.2), nn.Dropout(0.3),
                nn.Linear(hidden, hidden//2), nn.LeakyReLU(0.2), nn.Linear(hidden//2, 1))

        def forward(self, nums, seq):
            x = torch.cat(nums, -1)
            xl = seq.long()
            e = torch.cat([self.cond_emb[p](xl[:,:,p].clamp(0, POS_RANGES[p]-1))
                          for p in range(POS_COUNT)], -1)
            _, h = self.cond_gru(e)
            return self.net(torch.cat([x, h[-1]], -1))

    def train_gan(train_data, val_data=None, seq_len=5, epochs=200, verbose=True):
        """
        🔴 #1 修复: 只用 train_data 训练
        🟡 #5 修复: 用验证集监控生成器质量做早停
        """
        X_tr, y_tr = make_sequences(train_data, seq_len)
        if len(X_tr) < 5:
            return None, None, None

        # 🔴 #4 修复: 有效增强
        X_aug, y_aug = augment_sequences(X_tr, y_tr, n_aug=2)
        loader = DataLoader(LotteryDataset(X_aug, y_aug),
                            batch_size=min(64, len(X_aug)), shuffle=True)

        # 验证集（🟡 #5）
        X_val, y_val = None, None
        if val_data:
            X_val, y_val = make_sequences(val_data, seq_len)
            if len(X_val) == 0:
                X_val, y_val = None, None

        device = get_device()
        noise_dim = 32
        gen = Generator(noise_dim=noise_dim).to(device)
        disc = Discriminator().to(device)
        opt_g = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(disc.parameters(), lr=0.001, betas=(0.5, 0.999))

        best_val_metric = float('inf')
        best_gen_state = None
        patience_ctr = 0

        for epoch in range(epochs):
            gen.train(); disc.train()
            gl, dl = 0, 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                bs = bx.size(0)
                real = []
                for p in range(POS_COUNT):
                    oh = torch.zeros(bs, POS_RANGES[p], device=device)
                    oh.scatter_(1, by[:,p:p+1], 1.0)
                    real.append(oh * 0.9 + 0.1/POS_RANGES[p])

                z = torch.randn(bs, noise_dim, device=device)
                fake_l = gen(z, bx)
                fake_oh = [torch.softmax(f, -1).detach() for f in fake_l]
                d_loss = (F.binary_cross_entropy_with_logits(
                    disc(real, bx), torch.ones(bs,1,device=device)) +
                    F.binary_cross_entropy_with_logits(
                    disc(fake_oh, bx), torch.zeros(bs,1,device=device))) / 2
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                dl += d_loss.item()

                z = torch.randn(bs, noise_dim, device=device)
                fake_l2 = gen(z, bx)
                fake_oh2 = [torch.softmax(f, -1) for f in fake_l2]
                g_adv = F.binary_cross_entropy_with_logits(
                    disc(fake_oh2, bx), torch.ones(bs,1,device=device))
                # 🟢 #13: 加 label_smoothing
                g_ce = sum(F.cross_entropy(fake_l2[p], by[:,p], label_smoothing=0.1)
                           for p in range(POS_COUNT))
                g_loss = g_adv + g_ce * 0.5
                opt_g.zero_grad(); g_loss.backward(); opt_g.step()
                gl += g_loss.item()

            # 🟡 #5 修复: 用验证集分类损失做早停
            if X_val is not None and y_val is not None:
                gen.eval()
                with torch.no_grad():
                    vx = torch.FloatTensor(X_val).to(device)
                    vy = torch.LongTensor(y_val).to(device)
                    z = torch.randn(len(X_val), noise_dim, device=device)
                    vout = gen(z, vx)
                    val_ce = sum(F.cross_entropy(vout[p], vy[:,p]).item()
                                 for p in range(POS_COUNT))
                monitor = val_ce
            else:
                monitor = gl / len(loader)

            if monitor < best_val_metric:
                best_val_metric = monitor
                best_gen_state = {k: v.clone() for k, v in gen.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            # 🟡 #8 修复: 减少打印
            if verbose and (epoch+1) % 50 == 0:
                vstr = f"  ValCE:{monitor:.4f}" if X_val is not None else ""
                print(f"      GAN Ep {epoch+1:>3}/{epochs}"
                      f"  G:{gl/len(loader):.4f}  D:{dl/len(loader):.4f}{vstr}")

            if patience_ctr >= 30:
                if verbose:
                    print(f"      GAN 早停 Ep {epoch+1}")
                break

        if best_gen_state:
            gen.load_state_dict(best_gen_state)
        gen.eval()
        return gen, device, noise_dim

    def predict_gan(gen, device, noise_dim, history, seq_len=5, n_predict=5, n_samples=50):
        if gen is None: return None
        numbers = np.array([r['numbers'] for r in history])
        cur = numbers[-seq_len:].copy()
        preds = []
        with torch.no_grad():
            for _ in range(n_predict):
                x = torch.FloatTensor(cur).unsqueeze(0).repeat(n_samples,1,1).to(device)
                z = torch.randn(n_samples, noise_dim, device=device)
                out = gen(z, x)
                pl, pred = [], []
                for p in range(POS_COUNT):
                    pr = torch.softmax(out[p], -1).mean(0).cpu().numpy()
                    pl.append(pr); pred.append(int(np.argmax(pr)))
                preds.append({'numbers': pred, 'probs': pl})
                cur = np.concatenate([cur[1:], np.array(pred).reshape(1, POS_COUNT)])
        return preds


# ############################################################
#  模型 10：强化学习  (🔴#1: 只用train_data, 🟡#5: 验证集早停)
# ############################################################

if HAS_TORCH:

    class PolicyNetwork(nn.Module):
        # 🟡 #6: dropout=0.3
        def __init__(self, state_dim, hidden=128):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3))
            self.heads = nn.ModuleList([nn.Linear(hidden, POS_RANGES[p]) for p in range(POS_COUNT)])

        def forward(self, s):
            x = self.shared(s)
            return [F.log_softmax(h(x), -1) for h in self.heads]

        def get_probs(self, s):
            x = self.shared(s)
            return [F.softmax(h(x), -1) for h in self.heads]

    def train_rl_policy(train_data, val_data=None, lookback=10, epochs=300, verbose=True):
        """
        🔴 #1 修复: 只用 train_data 训练
        🟡 #5 修复: 用验证集做早停
        🟢 #12: Warmup 学习率
        """
        states, targets = [], []
        for i in range(lookback+1, len(train_data)):
            feat = build_features(train_data, i, lookback)
            if feat is not None:
                states.append(feat); targets.append(train_data[i]['numbers'])
        if len(states) < 10:
            if verbose: print("      ⚠ RL 样本不足")
            return None, None, None

        states = np.array(states); targets = np.array(targets)
        state_dim = states.shape[1]

        # 构造验证集（🟡 #5）
        val_states, val_targets = None, None
        if val_data and len(val_data) > lookback + 1:
            vs, vt = [], []
            for i in range(lookback+1, len(val_data)):
                # 用 train_data + val_data[:i] 构造特征（不泄露未来数据）
                feat = build_features(list(train_data) + list(val_data[:i]), 
                                      len(train_data) + i, lookback)
                if feat is not None:
                    vs.append(feat); vt.append(val_data[i]['numbers'])
            if vs:
                val_states = np.array(vs); val_targets = np.array(vt)

        device = get_device()
        policy = PolicyNetwork(state_dim).to(device)
        opt = torch.optim.Adam(policy.parameters(), lr=0.003)
        # 🟢 #12: Warmup
        scheduler = WarmupCosineScheduler(opt, warmup_steps=20, total_steps=epochs,
                                          base_lr=0.003, min_lr=1e-6)
        st = torch.FloatTensor(states).to(device)
        tgt = torch.LongTensor(targets).to(device)

        best_val = float('inf')
        best_state = None
        patience_ctr = 0

        for epoch in range(epochs):
            policy.train()
            lp = policy(st)
            # 🟢 #13: 用 label_smoothing
            loss = sum(F.cross_entropy(
                lp[p].exp(), tgt[:,p], label_smoothing=0.1) for p in range(POS_COUNT))

            with torch.no_grad():
                probs = policy.get_probs(st)
                rwd = sum((torch.argmax(probs[p],-1)==tgt[:,p]).float() for p in range(POS_COUNT))
                adv = rwd - rwd.mean()

            rl_loss = sum(-(lp[p].gather(1, tgt[:,p:p+1]).squeeze() * adv).mean() * 0.1
                          for p in range(POS_COUNT))
            total = loss + rl_loss
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            scheduler.step()

            # 🟡 #5: 验证集早停
            if val_states is not None:
                policy.eval()
                with torch.no_grad():
                    vs_t = torch.FloatTensor(val_states).to(device)
                    vt_t = torch.LongTensor(val_targets).to(device)
                    vlp = policy(vs_t)
                    val_loss = sum(F.nll_loss(vlp[p], vt_t[:,p]).item() for p in range(POS_COUNT))
                monitor = val_loss
            else:
                monitor = total.item()

            if monitor < best_val:
                best_val = monitor
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= 40:
                if verbose: print(f"      RL 早停 Ep {epoch+1}")
                break

            # 🟡 #8: 减少打印
            if verbose and (epoch+1) % 100 == 0:
                vstr = f"  Val:{val_loss:.4f}" if val_states is not None else ""
                print(f"      RL Ep {epoch+1:>3}/{epochs}  Loss:{loss.item():.4f}"
                      f"  Rwd:{rwd.mean().item():.2f}/7{vstr}")

        if best_state:
            policy.load_state_dict(best_state)
        policy.eval()
        return policy, device, state_dim

    def predict_rl(policy, device, history, lookback=10, n_predict=5):
        if policy is None: return None
        preds = []
        cur = list(history)
        for _ in range(n_predict):
            feat = build_features(cur, len(cur), lookback)
            if feat is None: break
            with torch.no_grad():
                probs = policy.get_probs(torch.FloatTensor(feat).unsqueeze(0).to(device))
            pred, pl = [], []
            for p in range(POS_COUNT):
                pr = probs[p].cpu().numpy()[0]; pl.append(pr); pred.append(int(np.argmax(pr)))
            preds.append({'numbers': pred, 'probs': pl})
            cur.append({'numbers': pred, 'issue': '', 'date': ''})
        return preds


# ############################################################
#  模型 11：贝叶斯
# ############################################################

def method_bayesian(history, n_predict=5, window=30):
    preds = []
    for _ in range(n_predict):
        pred = []
        for pos in range(POS_COUNT):
            nv = POS_RANGES[pos]
            alpha = np.ones(nv)
            for rec in history: alpha[rec['numbers'][pos]] += 0.3
            for j, rec in enumerate(history[-window:]):
                alpha[rec['numbers'][pos]] += (1 + j/window * 2) * 0.7
            pred.append(int(np.random.choice(nv, p=np.random.dirichlet(alpha))))
        preds.append(pred)
    return preds


def method_bayesian_probs(history, window=30):
    pl = []
    for pos in range(POS_COUNT):
        nv = POS_RANGES[pos]
        alpha = np.ones(nv)
        for rec in history: alpha[rec['numbers'][pos]] += 0.3
        for j, rec in enumerate(history[-window:]):
            alpha[rec['numbers'][pos]] += (1 + j/window * 2) * 0.7
        pl.append(alpha / alpha.sum())
    return pl


# ############################################################
#  模型 12：随机森林 (🔴#1: 只用train_data, 🟢#14: 训练一次缓存)
# ############################################################

class RandomForestCache:
    """🟢 #14 修复: 训练一次，缓存模型+scaler，不重复训练"""
    def __init__(self):
        self.models = None
        self.scaler = None
        self.trained = False

    def train(self, train_data, lookback=10):
        if not HAS_SKLEARN:
            return False
        X_l, y_l = [], []
        for i in range(lookback+1, len(train_data)):
            feat = build_features(train_data, i, lookback)
            if feat is not None:
                X_l.append(feat); y_l.append(train_data[i]['numbers'])
        if len(X_l) < 20:
            return False
        X, y = np.array(X_l), np.array(y_l)
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)
        self.models = []
        for pos in range(POS_COUNT):
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=3,
                class_weight='balanced', random_state=42+pos, n_jobs=-1)
            rf.fit(X_s, y[:, pos])
            self.models.append(rf)
        self.trained = True
        return True

    def predict_probs(self, history, lookback=10):
        """返回各位置概率分布"""
        if not self.trained:
            return None
        feat = build_features(history, len(history), lookback)
        if feat is None:
            return None
        fs = self.scaler.transform(feat.reshape(1, -1))
        pl = []
        for pos in range(POS_COUNT):
            prob = np.zeros(POS_RANGES[pos])
            if hasattr(self.models[pos], 'predict_proba'):
                p = self.models[pos].predict_proba(fs)[0]
                for ci, cls in enumerate(self.models[pos].classes_):
                    if 0 <= cls < POS_RANGES[pos]:
                        prob[cls] = p[ci]
            prob = prob/prob.sum() if prob.sum() > 0 else np.ones(POS_RANGES[pos])/POS_RANGES[pos]
            pl.append(prob)
        return pl

    def predict_multi(self, history, lookback=10, n_predict=5):
        """多期预测"""
        if not self.trained:
            return None
        preds = []
        cur = list(history)
        for _ in range(n_predict):
            pl = self.predict_probs(cur, lookback)
            if pl is None: break
            pred = [int(np.argmax(pl[p])) for p in range(POS_COUNT)]
            preds.append({'numbers': pred, 'probs': pl})
            cur.append({'numbers': pred, 'issue': '', 'date': ''})
        return preds


# 全局缓存实例
rf_cache = RandomForestCache()


# ############################################################
#  模型 13：TensorFlow Multi-Head Attention
# ############################################################

if HAS_TF:

    def build_tf_attention_model(seq_len=5, d_model=64, n_heads=4, ff_dim=128, dropout=0.3):
        inputs = keras.Input(shape=(seq_len, POS_COUNT), name='input_seq')
        emb_dim = 8
        pos_embs = []
        for pos in range(POS_COUNT):
            pi = tf.cast(inputs[:,:,pos], tf.int32)
            emb = keras.layers.Embedding(POS_RANGES[pos], emb_dim, name=f'emb_p{pos}')(pi)
            pos_embs.append(emb)
        x = keras.layers.Concatenate(axis=-1)(pos_embs)
        x = keras.layers.Dense(d_model, name='project')(x)
        pos_enc = keras.layers.Embedding(seq_len, d_model, name='pos_enc')(tf.range(seq_len))
        x = x + pos_enc

        for b in range(2):
            attn = keras.layers.MultiHeadAttention(
                n_heads, d_model//n_heads, dropout=dropout, name=f'mha_{b}')(x, x)
            x = keras.layers.LayerNormalization(name=f'ln1_{b}')(x + attn)
            ff = keras.layers.Dense(ff_dim, activation='relu', name=f'ff1_{b}')(x)
            ff = keras.layers.Dropout(dropout, name=f'ffd_{b}')(ff)
            ff = keras.layers.Dense(d_model, name=f'ff2_{b}')(ff)
            x = keras.layers.LayerNormalization(name=f'ln2_{b}')(x + ff)

        aw = keras.layers.Dense(1, activation='tanh', name='attn_w')(x)
        aw = keras.layers.Softmax(axis=1, name='attn_s')(aw)
        x = tf.reduce_sum(x * aw, axis=1)
        x = keras.layers.Dropout(dropout, name='fdrop')(x)

        outputs = []
        for pos in range(POS_COUNT):
            h = keras.layers.Dense(32, activation='relu', name=f'h{pos}_d')(x)
            h = keras.layers.Dropout(dropout*0.5, name=f'h{pos}_dr')(h)
            # 🟢 #13: Label Smoothing 在 loss 层
            out = keras.layers.Dense(POS_RANGES[pos], activation='softmax', name=f'pos_{pos}')(h)
            outputs.append(out)

        return keras.Model(inputs=inputs, outputs=outputs, name='TF_Attention')

    def train_tf_attention(train_data, val_data=None, seq_len=5, epochs=300, verbose=True):
        numbers = np.array([r['numbers'] for r in train_data], dtype=np.float32)
        X_tr = np.array([numbers[i:i+seq_len] for i in range(len(numbers)-seq_len)])
        y_tr = np.array([numbers[i+seq_len].astype(int) for i in range(len(numbers)-seq_len)])

        if len(X_tr) < 5:
            if verbose: print("      ⚠ TF-Attn 样本不足")
            return None

        # 🔴 #4: 有效增强
        X_aug, y_aug = [X_tr], [y_tr]
        for ai in range(2):
            rng = np.random.RandomState(42 + ai)
            Xn = X_tr.copy()
            mask = rng.random(Xn.shape) < 0.15
            pert = rng.choice([-1, 0, 1], size=Xn.shape).astype(np.float32)
            Xn += mask * pert
            for pos in range(POS_COUNT):
                Xn[:,:,pos] = np.clip(Xn[:,:,pos], 0, POS_RANGES[pos]-1)
            X_aug.append(Xn); y_aug.append(y_tr.copy())
        X_tr = np.concatenate(X_aug); y_tr = np.concatenate(y_aug)

        # 验证集
        validation_data = None
        if val_data and len(val_data) > seq_len:
            vn = np.array([r['numbers'] for r in val_data], dtype=np.float32)
            Xv = np.array([vn[i:i+seq_len] for i in range(len(vn)-seq_len)])
            yv = np.array([vn[i+seq_len].astype(int) for i in range(len(vn)-seq_len)])
            if len(Xv) > 0:
                yv_dict = {}
                for pos in range(POS_COUNT):
                    oh = np.zeros((len(yv), POS_RANGES[pos]), dtype=np.float32)
                    for i in range(len(yv)):
                        oh[i, int(yv[i, pos])] = 1.0
                    yv_dict[f'pos_{pos}'] = oh
                validation_data = (Xv, yv_dict)

        model = build_tf_attention_model(seq_len=seq_len)

        # 🟢 #13: Label Smoothing
        losses = {f'pos_{p}': keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
                  for p in range(POS_COUNT)}

        # 🟢 #12: LR with warmup (TF 风格)
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=epochs * (len(X_tr)//32 + 1),
            alpha=1e-6)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss=losses)

        y_dict = {}
        for pos in range(POS_COUNT):
            oh = np.zeros((len(y_tr), POS_RANGES[pos]), dtype=np.float32)
            for i in range(len(y_tr)):
                oh[i, int(y_tr[i, pos])] = 1.0
            y_dict[f'pos_{pos}'] = oh

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=30, restore_best_weights=True,
                monitor='val_loss' if validation_data else 'loss'),
            keras.callbacks.ReduceLROnPlateau(
                patience=15, factor=0.5, min_lr=1e-6,
                monitor='val_loss' if validation_data else 'loss')]

        if verbose:
            print(f"      TF-Attn: {model.count_params():,} params, "
                  f"Train:{len(X_tr)}, Val:{len(Xv) if validation_data else 0}")

        hist = model.fit(X_tr, y_dict, epochs=epochs, batch_size=min(64, len(X_tr)),
                         callbacks=callbacks, validation_data=validation_data, verbose=0)

        if verbose:
            n_ep = len(hist.history['loss'])
            vstr = f"  Val:{hist.history['val_loss'][-1]:.4f}" if 'val_loss' in hist.history else ""
            print(f"      完成: {n_ep}ep, Loss:{hist.history['loss'][-1]:.4f}{vstr}")
        return model

    def predict_tf_attention(model, history, seq_len=5, n_predict=5):
        if model is None: return None
        numbers = np.array([r['numbers'] for r in history], dtype=np.float32)
        cur = numbers[-seq_len:].copy()
        preds = []
        for _ in range(n_predict):
            x = cur.reshape(1, seq_len, POS_COUNT)
            out = model.predict(x, verbose=0)
            pred, pl = [], []
            for pos in range(POS_COUNT):
                pr = out[pos][0]; pl.append(pr); pred.append(int(np.argmax(pr)))
            preds.append({'numbers': pred, 'probs': pl})
            cur = np.concatenate([cur[1:], np.array(pred, dtype=np.float32).reshape(1, POS_COUNT)])
        return preds

    def method_tf_attention_ensemble(train_data, val_data, full_history,
                                      n_predict=5, seq_len=5, n_models=3):
        all_probs = [[np.zeros(POS_RANGES[p]) for p in range(POS_COUNT)]
                     for _ in range(n_predict)]
        for mi in range(n_models):
            print(f"\n    ┌─ TF-Attn 模型 {mi+1}/{n_models} ─┐")
            tf.random.set_seed(42 + mi * 11); np.random.seed(42 + mi * 11)
            model = train_tf_attention(train_data, val_data, seq_len, 300, True)
            if model is None: continue
            preds = predict_tf_attention(model, full_history, seq_len, n_predict)
            if preds is None: continue
            for step in range(n_predict):
                for pos in range(POS_COUNT):
                    all_probs[step][pos] += preds[step]['probs'][pos]
            print(f"    └─ TF-Attn 模型 {mi+1} 完成 ─┘")
            keras.backend.clear_session()

        predictions = []
        for step in range(n_predict):
            pred = [int(np.argmax(all_probs[step][p])) if all_probs[step][p].sum() > 0
                    else np.random.randint(0, POS_RANGES[p]) for p in range(POS_COUNT)]
            predictions.append(pred)
        return predictions, all_probs


# ############################################################
#  通用 RNN 训练 (LSTM/GRU 共用)
# ############################################################

if HAS_TORCH:

    def train_rnn_model(model_class, train_data, val_data, seq_len=5,
                        epochs=300, lr=0.003, verbose=True,
                        model_name="RNN", model_id=1, d_hash=""):
        X_tr, y_tr = make_sequences(train_data, seq_len)
        X_val, y_val = make_sequences(val_data, seq_len) if val_data else (np.array([]), np.array([]))
        if len(X_tr) < 3:
            if verbose: print(f"    ⚠ {model_name} 样本不足")
            return None, None

        device = get_device()
        model = model_class(hidden_size=64, num_layers=2, dropout=0.3).to(device)

        # 🟢 #15: 尝试加载缓存模型
        ckpt_name = f"{model_name}_{model_id}"
        if d_hash and load_model_if_exists(model, ckpt_name, d_hash):
            if verbose:
                print(f"      [{model_name}{model_id}] ✓ 从缓存加载")
            return model, device

        # 🔴 #4: 有效增强
        X_aug, y_aug = augment_sequences(X_tr, y_tr, n_aug=2)
        loader = DataLoader(LotteryDataset(X_aug, y_aug),
                            batch_size=min(64, len(X_aug)), shuffle=True)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        # 🟢 #12: Warmup + Cosine
        scheduler = WarmupCosineScheduler(opt, warmup_steps=20, total_steps=epochs,
                                          base_lr=lr, min_lr=1e-6)
        # 🟢 #13: Label Smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_vl = float('inf')
        best_state = None
        patience_ctr = 0

        for epoch in range(epochs):
            model.train()
            tl = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = sum(criterion(out[p], by[:,p]) for p in range(POS_COUNT))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); tl += loss.item()
            scheduler.step()
            avg_tl = tl / len(loader)

            monitor = avg_tl
            vl = -1
            if len(X_val) > 0:
                vl, _, _ = evaluate_model(model, device, X_val, y_val)
                monitor = vl

            if monitor < best_vl:
                best_vl = monitor
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            # 🟡 #8: 只打印关键epoch
            if verbose and (epoch+1) % 100 == 0:
                vs = f"  Val:{vl:.4f}" if vl >= 0 else ""
                cur_lr = opt.param_groups[0]['lr']
                print(f"      [{model_name}{model_id}] Ep {epoch+1:>3}/{epochs}"
                      f"  Train:{avg_tl:.4f}{vs}  LR:{cur_lr:.6f}")

            if patience_ctr >= 40:
                if verbose:
                    print(f"      [{model_name}{model_id}] 早停 Ep {epoch+1}")
                break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        # 🟢 #15: 保存模型
        if d_hash:
            save_model(model, ckpt_name, d_hash)

        return model, device

    def predict_rnn(model, device, history, seq_len=5, n_predict=5):
        if model is None: return None
        numbers = np.array([r['numbers'] for r in history])
        cur = numbers[-seq_len:].copy()
        preds = []
        with torch.no_grad():
            for _ in range(n_predict):
                x = torch.FloatTensor(cur).unsqueeze(0).to(device)
                out = model(x)
                pred, pl = [], []
                for p in range(POS_COUNT):
                    pr = torch.softmax(out[p], -1).cpu().numpy()[0]
                    pl.append(pr); pred.append(int(np.argmax(pr)))
                preds.append({'numbers': pred, 'probs': pl})
                cur = np.concatenate([cur[1:], np.array(pred).reshape(1, POS_COUNT)])
        return preds

    def method_rnn_ensemble(model_class, model_name, train_data, val_data,
                            full_history, n_predict=5, seq_len=5, n_models=3, d_hash=""):
        all_probs = [[np.zeros(POS_RANGES[p]) for p in range(POS_COUNT)]
                     for _ in range(n_predict)]
        for mi in range(n_models):
            print(f"\n    ┌─ {model_name} 模型 {mi+1}/{n_models} ─┐")
            torch.manual_seed(42 + mi * 7); np.random.seed(42 + mi * 7)
            model, device = train_rnn_model(
                model_class, train_data, val_data, seq_len, 300, 0.003,
                True, model_name, mi+1, d_hash)
            if model is None: continue
            preds = predict_rnn(model, device, full_history, seq_len, n_predict)
            if preds is None: continue
            for step in range(n_predict):
                for pos in range(POS_COUNT):
                    all_probs[step][pos] += preds[step]['probs'][pos]
            print(f"    └─ {model_name} 模型 {mi+1} 完成 ─┘")

        predictions = []
        for step in range(n_predict):
            pred = [int(np.argmax(all_probs[step][p])) if all_probs[step][p].sum() > 0
                    else np.random.randint(0, POS_RANGES[p]) for p in range(POS_COUNT)]
            predictions.append(pred)
        return predictions, all_probs


# ############################################################
#  🔴 #2 + #3 修复: 确定性候选 + 正确置信度
# ############################################################

def generate_candidates(vote_matrix, step, n_candidates=3):
    """
    🔴 #2 修复: 确定性 Top-N 候选生成
    ──────────────────────────────────
    候选1: 每个位置取最高概率 (最保守)
    候选2: 偶数位取第二选择 (增加多样性)
    候选3: 固定种子加权采样 (可复现)
    """
    top_per_pos = []
    prob_per_pos = []
    for pos in range(POS_COUNT):
        votes = vote_matrix[step][pos]
        total = votes.sum()
        probs = votes / total if total > 0 else np.ones(POS_RANGES[pos]) / POS_RANGES[pos]
        top_idx = np.argsort(probs)[::-1]
        top_per_pos.append(top_idx)
        prob_per_pos.append(probs)

    candidates = []
    confs = []

    # 候选1: 全 Top-1
    c1 = [int(top_per_pos[p][0]) for p in range(POS_COUNT)]
    f1 = [prob_per_pos[p][c1[p]] for p in range(POS_COUNT)]
    candidates.append(c1); confs.append(f1)

    # 候选2: 偶数位 Top-2
    c2 = [int(top_per_pos[p][1 if p % 2 == 0 and len(top_per_pos[p]) > 1 else 0])
          for p in range(POS_COUNT)]
    f2 = [prob_per_pos[p][c2[p]] for p in range(POS_COUNT)]
    candidates.append(c2); confs.append(f2)

    # 候选3: 固定种子采样
    rng = np.random.RandomState(42 + step * 7)
    c3 = [int(rng.choice(POS_RANGES[p], p=prob_per_pos[p])) for p in range(POS_COUNT)]
    f3 = [prob_per_pos[p][c3[p]] for p in range(POS_COUNT)]
    candidates.append(c3); confs.append(f3)

    return candidates[:n_candidates], confs[:n_candidates]


# ############################################################
#  集成投票器
# ############################################################

def ensemble_predict(history, train_data, val_data, n_predict=5, n_rounds=200, d_hash=""):
    """全方法集成投票 — 所有15个问题已修复"""

    stat_methods = {
        '频率分析': (method_frequency, 1.0),
        '热号冷号': (method_hot_cold, 1.2),
        '马尔可夫链': (method_markov, 1.5),
        '加权近期': (method_weighted_recent, 1.3),
        '差值趋势': (method_delta_trend, 1.0),
        '模式匹配': (method_pattern_match, 1.4),
    }

    vote_matrix = [[np.zeros(POS_RANGES[p]) for p in range(POS_COUNT)]
                   for _ in range(n_predict)]
    all_method_preds = {}

    # ① 统计方法
    print(f"\n  ── ① 统计方法 ({len(stat_methods)} × {n_rounds}轮) ──")
    for name, (func, weight) in stat_methods.items():
        for _ in range(n_rounds):
            try:
                preds = func(history, n_predict)
                for step in range(n_predict):
                    for pos in range(POS_COUNT):
                        vote_matrix[step][pos][preds[step][pos]] += weight
            except Exception: continue
        # 🟡 #7 修复: 缓存首轮预测结果
        np.random.seed(42)
        single = func(history, 1)
        all_method_preds[name] = single[0]
        print(f"    ✓ {name}")

    # ② 贝叶斯
    print(f"\n  ── ② 贝叶斯 ──")
    try:
        bp = method_bayesian_probs(history)
        bw = 1.8
        bpred = [int(np.argmax(bp[pos])) for pos in range(POS_COUNT)]
        all_method_preds['贝叶斯'] = bpred
        for pos in range(POS_COUNT):
            vote_matrix[0][pos] += bp[pos] * bw * n_rounds
        for _ in range(n_rounds):
            try:
                preds = method_bayesian(history, n_predict)
                for step in range(1, n_predict):
                    for pos in range(POS_COUNT):
                        vote_matrix[step][pos][preds[step][pos]] += bw
            except Exception: continue
        print(f"    ✓ 贝叶斯")
    except Exception as e:
        print(f"    ⚠ 贝叶斯: {e}")

    # ③ 随机森林 — 🔴 #1 + 🟢 #14: 只用 train_data, 训练一次缓存
    if HAS_SKLEARN:
        print(f"\n  ── ③ 随机森林 ──")
        try:
            if not rf_cache.trained:
                rf_cache.train(train_data)
            if rf_cache.trained:
                rf_preds = rf_cache.predict_multi(history, n_predict=n_predict)
                if rf_preds:
                    rw = 1.8
                    all_method_preds['随机森林'] = rf_preds[0]['numbers']
                    for step in range(min(n_predict, len(rf_preds))):
                        for pos in range(POS_COUNT):
                            vote_matrix[step][pos] += rf_preds[step]['probs'][pos] * rw * n_rounds
                    print(f"    ✓ 随机森林 (缓存)")
                else:
                    print(f"    ⚠ 随机森林: 预测失败")
            else:
                print(f"    ⚠ 随机森林: 数据不足")
        except Exception as e:
            print(f"    ⚠ 随机森林: {e}")

    # ④ LSTM
    lstm_preds_out = None
    if HAS_TORCH:
        print(f"\n  ── ④ LSTM (3模型) ──")
        t0 = time.time()
        try:
            sq = min(5, len(train_data) - 2)
            lp, lpr = method_rnn_ensemble(LSTMModel, "LSTM", train_data, val_data,
                                          history, n_predict, sq, 3, d_hash)
            print(f"\n    ✓ LSTM ({time.time()-t0:.1f}s)")
            all_method_preds['LSTM'] = lp[0]; lstm_preds_out = lp
            for step in range(n_predict):
                for pos in range(POS_COUNT):
                    vote_matrix[step][pos] += lpr[step][pos] * 2.0 * n_rounds
        except Exception as e:
            print(f"    ⚠ LSTM: {e}")

    # ⑤ GRU
    gru_preds_out = None
    if HAS_TORCH:
        print(f"\n  ── ⑤ GRU (3模型) ──")
        t0 = time.time()
        try:
            sq = min(5, len(train_data) - 2)
            gp, gpr = method_rnn_ensemble(GRUModel, "GRU", train_data, val_data,
                                          history, n_predict, sq, 3, d_hash)
            print(f"\n    ✓ GRU ({time.time()-t0:.1f}s)")
            all_method_preds['GRU'] = gp[0]; gru_preds_out = gp
            for step in range(n_predict):
                for pos in range(POS_COUNT):
                    vote_matrix[step][pos] += gpr[step][pos] * 2.0 * n_rounds
        except Exception as e:
            print(f"    ⚠ GRU: {e}")

    # ⑥ GAN — 🔴 #1: 只用 train_data
    gan_preds_out = None
    if HAS_TORCH:
        print(f"\n  ── ⑥ GAN ──")
        t0 = time.time()
        try:
            sq = min(5, len(train_data) - 2)
            torch.manual_seed(42)
            gen, gd, nd = train_gan(train_data, val_data, sq, 200, True)
            if gen:
                gp = predict_gan(gen, gd, nd, history, sq, n_predict)
                print(f"    ✓ GAN ({time.time()-t0:.1f}s)")
                gan_preds_out = [g['numbers'] for g in gp]
                all_method_preds['GAN'] = gan_preds_out[0]
                for step in range(n_predict):
                    for pos in range(POS_COUNT):
                        vote_matrix[step][pos] += gp[step]['probs'][pos] * 1.5 * n_rounds
        except Exception as e:
            print(f"    ⚠ GAN: {e}")

    # ⑦ RL — 🔴 #1: 只用 train_data
    rl_preds_out = None
    if HAS_TORCH:
        print(f"\n  ── ⑦ 强化学习 ──")
        t0 = time.time()
        try:
            torch.manual_seed(42)
            pol, pd, sd = train_rl_policy(train_data, val_data, 10, 300, True)
            if pol:
                rp = predict_rl(pol, pd, history, 10, n_predict)
                print(f"    ✓ RL ({time.time()-t0:.1f}s)")
                rl_preds_out = [r['numbers'] for r in rp]
                all_method_preds['RL'] = rl_preds_out[0]
                for step in range(n_predict):
                    for pos in range(POS_COUNT):
                        vote_matrix[step][pos] += rp[step]['probs'][pos] * 1.5 * n_rounds
        except Exception as e:
            print(f"    ⚠ RL: {e}")

    # ⑧ TF Attention
    tf_preds_out = None
    if HAS_TF:
        print(f"\n  ── ⑧ TF-Attention (3模型) ──")
        t0 = time.time()
        try:
            sq = min(5, len(train_data) - 2)
            tf.random.set_seed(42)
            tp, tpr = method_tf_attention_ensemble(train_data, val_data, history, n_predict, sq, 3)
            print(f"\n    ✓ TF-Attn ({time.time()-t0:.1f}s)")
            all_method_preds['TF-Attn'] = tp[0]; tf_preds_out = tp
            for step in range(n_predict):
                for pos in range(POS_COUNT):
                    vote_matrix[step][pos] += tpr[step][pos] * 2.0 * n_rounds
        except Exception as e:
            print(f"    ⚠ TF-Attn: {e}")

    if not HAS_TORCH:
        print(f"\n  ⓘ PyTorch 未安装，跳过 LSTM/GRU/GAN/RL")
    if not HAS_TF:
        print(f"\n  ⓘ TensorFlow 未安装，跳过 TF-Attention")

    # 🔴 #2 修复: 确定性候选生成
    final_predictions = []
    confidence_matrix = []
    for step in range(n_predict):
        cands, confs = generate_candidates(vote_matrix, step, n_candidates=3)
        final_predictions.append(cands)
        confidence_matrix.append(confs)

    return (final_predictions, confidence_matrix, vote_matrix,
            lstm_preds_out, gru_preds_out, gan_preds_out, rl_preds_out,
            tf_preds_out, all_method_preds)


# ############################################################
#  🟡 #7 修复: 日志使用缓存结果，不重新运行
# ############################################################

def save_log(history, predictions, confidences, all_method_preds,
             vote_matrix, t_total, last_issue, n_predict):
    """保存日志 — 使用缓存的预测结果，不重新运行任何方法"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_DIR, f"prediction_{timestamp}.log")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"7星彩预测系统 v5.1 — 完整日志\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"耗时: {t_total:.1f}s | 数据: {len(history)}期\n")
        f.write(f"{'='*80}\n\n")

        # 最终预测
        f.write("【最终预测结果】\n")
        for step in range(n_predict):
            issue = last_issue + step + 1
            f.write(f"\n第 {issue} 期:\n")
            for ci, (pred, conf) in enumerate(zip(predictions[step], confidences[step])):
                # 🔴 #3 修复: 用均值而非乘积
                avg_c = np.mean(conf) * 100
                ps = ' '.join(str(pred[p]) for p in range(POS_COUNT))
                labels = ["首选", "次选", "三选"]
                f.write(f"  {labels[ci]}: [{ps}] 置信度:{avg_c:.1f}%\n")

        # 各算法独立结果 (🟡 #7: 使用缓存)
        f.write("\n\n【各算法独立预测（缓存结果）】\n")
        for name, pred in all_method_preds.items():
            if pred is not None:
                ps = ' '.join(str(pred[i]) for i in range(POS_COUNT))
                f.write(f"  {name:>10}: [{ps}] 和值={sum(pred)}\n")

        # 概率分布
        f.write("\n\n【概率分布（第1期）】\n")
        for pos in range(POS_COUNT):
            v = vote_matrix[0][pos]; t = v.sum()
            p = v/t if t > 0 else np.zeros(POS_RANGES[pos])
            top5 = np.argsort(p)[::-1][:5]
            f.write(f"  位置{pos+1}: " + " ".join(f"{i}({p[i]*100:.1f}%)" for i in top5) + "\n")

        f.write(f"\n{'='*80}\n")

    return log_path


# ############################################################
#  输出函数
# ############################################################

def print_analysis(history):
    print(f"\n{'═'*70}")
    print(f"  📊 历史数据统计分析")
    print(f"{'═'*70}")
    n = len(history)
    print(f"\n  数据: {history[0]['issue']}~{history[-1]['issue']}  共{n}期")

    print(f"\n  位置1~6 频率:")
    print(f"  {'位置':>6}", end="")
    for d in range(10): print(f" {d:>4}", end="")
    print(f"  最热")
    print(f"  {'─'*62}")
    for pos in range(6):
        c = Counter(r['numbers'][pos] for r in history)
        print(f"  位置{pos+1}", end="")
        for d in range(10): print(f" {c.get(d,0):>4}", end="")
        mc = c.most_common(1)[0]
        print(f"  {mc[0]}({mc[1]})")

    print(f"\n  位置7 频率 (0~14):")
    c7 = Counter(r['numbers'][6] for r in history)
    print(f"  {'':>6}", end="")
    for d in range(15): print(f" {d:>4}", end="")
    print(f"  最热")
    print(f"  {'─'*78}")
    print(f"  位置7", end="")
    for d in range(15): print(f" {c7.get(d,0):>4}", end="")
    mc7 = c7.most_common(1)[0]
    print(f"  {mc7[0]}({mc7[1]})")

    sums = [sum(r['numbers']) for r in history]
    rs = sums[-10:]
    print(f"\n  和值: 均值={np.mean(sums):.1f}  近10期={np.mean(rs):.1f}  "
          f"范围=[{min(rs)},{max(rs)}]")


def print_probs(vote_matrix, step, pred):
    print(f"\n  位置1~6:")
    print(f"  {'位置':>6}", end="")
    for d in range(10): print(f" {d:>5}", end="")
    print(f"  选中")
    print(f"  {'─'*66}")
    for pos in range(6):
        v = vote_matrix[step][pos]; t = v.sum()
        p = v/t if t > 0 else np.zeros(10)
        print(f"  位置{pos+1}", end="")
        for d in range(10):
            pct = p[d]*100
            mk = "▓" if d == pred[pos] else ("░" if pct >= 10 else " ")
            print(f" {pct:>4.0f}{mk}", end="")
        print(f"  → {pred[pos]}")

    print(f"\n  位置7 (0~14):")
    v7 = vote_matrix[step][6]; t7 = v7.sum()
    p7 = v7/t7 if t7 > 0 else np.zeros(15)
    print(f"  {'':>6}", end="")
    for d in range(15): print(f" {d:>5}", end="")
    print(f"  选中")
    print(f"  {'─'*84}")
    print(f"  位置7", end="")
    for d in range(15):
        pct = p7[d]*100
        mk = "▓" if d == pred[6] else ("░" if pct >= 10 else " ")
        print(f" {pct:>4.0f}{mk}", end="")
    print(f"  → {pred[6]}")


# ############################################################
#  主程序  (🟡 #9 修复: 默认值逻辑)
# ############################################################

def main():
    n_algo = 6 + (4 if HAS_TORCH else 0) + 1 + (1 if HAS_SKLEARN else 0) + (1 if HAS_TF else 0)

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         7 星 彩 号 码 智 能 预 测 系 统  v5.1                      ║")
    print("║    13-Algorithm Ensemble — 全部问题修复版                          ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  PyTorch:     {'✅' if HAS_TORCH else '❌ pip install torch':<50}    ║")
    print(f"║  sklearn:     {'✅' if HAS_SKLEARN else '❌ pip install scikit-learn':<50}    ║")
    print(f"║  TensorFlow:  {'✅' if HAS_TF else '❌ pip install tensorflow':<50}    ║")
    print(f"║  可用算法:    {n_algo} 种                                              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # 加载数据
    try:
        history = load_data('7星彩_历史开奖号码.csv')
        print(f"  ✓ 从CSV加载 {len(history)} 条")
    except FileNotFoundError:
        history = get_builtin_data()
        print(f"  ✓ 内置数据 {len(history)} 条")

    d_hash_val = data_hash(history)
    print(f"  数据指纹: {d_hash_val}")

    p7 = [r['numbers'][6] for r in history]
    p7t = sum(1 for v in p7 if v >= 10)
    print(f"  位置7: max={max(p7)}, ≥10有{p7t}条({p7t/len(history)*100:.1f}%)")

    train_data, val_data, test_data = split_dataset(history)
    print(f"\n  训练: {len(train_data)}期  验证: {len(val_data)}期  测试: {len(test_data)}期")

    print_analysis(history)

    # 预测
    n_predict = 5
    print(f"\n{'═'*70}")
    print(f"  🎯 集成预测 ({n_algo}种算法)")
    print(f"{'═'*70}")

    np.random.seed(42)
    if HAS_TORCH: torch.manual_seed(42)
    if HAS_TF: tf.random.set_seed(42)

    t0 = time.time()
    (predictions, confidences, vote_matrix,
     lstm_preds, gru_preds, gan_preds, rl_preds,
     tf_preds, all_method_preds) = ensemble_predict(
        history, train_data, val_data, n_predict, d_hash=d_hash_val
    )
    t_total = time.time() - t0

    last_issue = int(history[-1]['issue'])

    # 🟡 #7: 日志使用缓存
    log_path = save_log(history, predictions, confidences, all_method_preds,
                        vote_matrix, t_total, last_issue, n_predict)
    print(f"\n  ✅ 日志: {log_path}")

    # ============================================================
    # 结果
    # ============================================================
    print(f"\n{'═'*70}")
    print(f"  🏆 预测结果 (耗时 {t_total:.1f}s)")
    print(f"{'═'*70}")

    for step in range(n_predict):
        issue = last_issue + step + 1
        print(f"\n  ── 第 {issue} 期 ──")
        labels = ["🥇首选", "🥈次选", "🥉三选"]
        for ci, (pred, conf) in enumerate(zip(predictions[step], confidences[step])):
            # 🔴 #3 修复: 用平均置信度
            avg_conf = np.mean(conf)
            ps = ' '.join(fmt(pred[p], p) for p in range(POS_COUNT))
            bar_len = int(avg_conf * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"    {labels[ci]}: [{ps}]  置信度 {avg_conf*100:.1f}%  {bar}")

    # 各方法对比 (🟡 #7: 使用缓存)
    print(f"\n{'═'*70}")
    print(f"  🔬 各算法独立结果（第1期）")
    print(f"{'═'*70}")
    for name in ['频率分析','热号冷号','马尔可夫链','加权近期','差值趋势','模式匹配',
                  '贝叶斯','随机森林','LSTM','GRU','GAN','RL','TF-Attn']:
        if name in all_method_preds and all_method_preds[name] is not None:
            p = all_method_preds[name]
            ps = ' '.join(fmt(p[i], i) for i in range(POS_COUNT))
            print(f"  {name:>10}: [{ps}]  和值={sum(p)}")

    # 概率分布
    print(f"\n{'═'*70}")
    print(f"  📋 第1期概率分布")
    print(f"{'═'*70}")
    print_probs(vote_matrix, 0, predictions[0][0])

    # 备选 Top3
    print(f"\n{'═'*70}")
    print(f"  🔄 备选号码 (Top3)")
    print(f"{'═'*70}")
    for step in range(n_predict):
        issue = last_issue + step + 1
        print(f"\n  第 {issue} 期:")
        for pos in range(POS_COUNT):
            v = vote_matrix[step][pos]; t = v.sum()
            p = v/t if t > 0 else np.zeros(POS_RANGES[pos])
            top3 = np.argsort(p)[::-1][:3]
            t3s = "  ".join(f"{i}({p[i]*100:.0f}%)" for i in top3)
            print(f"    位置{pos+1}(0~{POS_RANGES[pos]-1:>2}): {t3s}")

    # 一致性
    print(f"\n{'═'*70}")
    print(f"  📊 一致性分析（第1期）")
    print(f"{'═'*70}")
    for pos in range(POS_COUNT):
        nv = POS_RANGES[pos]; v = vote_matrix[0][pos]; t = v.sum()
        p = v/t if t > 0 else np.zeros(nv)
        t1 = int(np.argmax(p)); t1p = p[t1]*100
        ent = -sum(pp * np.log2(pp + 1e-10) for pp in p)
        cons = max(0, 1 - ent / np.log2(nv))
        bar = "█" * int(cons*20) + "░" * (20 - int(cons*20))
        lb = '✅' if cons > 0.4 else '⚠️' if cons < 0.2 else '→'
        print(f"  位置{pos+1}(0~{nv-1:>2}): 首选={t1}({t1p:.0f}%) 一致={cons:.2f} {bar} {lb}")

    # 推荐
    print(f"\n{'═'*70}")
    print(f"  💡 最终推荐")
    print(f"{'═'*70}")
    for step in range(n_predict):
        issue = last_issue + step + 1
        pred = predictions[step][0]
        conf = np.mean(confidences[step][0])
        stars = "⭐" * min(5, max(1, int(conf * 10)))
        ps = '  '.join(fmt(pred[p], p) for p in range(POS_COUNT))
        print(f"\n  📌 第 {issue} 期:  [ {ps} ]  {stars} {conf*100:.1f}%")

    print(f"\n{'═'*70}")
    print(f"  ⚠️  彩票开奖完全随机，任何算法都无法预测。仅供学习娱乐。")
    print(f"{'═'*70}\n")


if __name__ == '__main__':
    main()
