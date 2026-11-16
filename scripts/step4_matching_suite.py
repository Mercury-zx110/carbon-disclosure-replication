# -*- coding: utf-8 -*-
"""
第四步（碳减排研究）：匹配方法全集
在 2020 年截面上实施：① 1:1 最近邻有放回（卡尺 0.05）② 1:1 无卡尺
③ 1:4 最近邻（权重=被匹配次数）④ 半径匹配（卡尺 0.05）
⑤ 核匹配（高斯核，带宽 0.05×SD(logitPS)）⑥ 马氏距离 1:1
⑦ IPW-ATT（倾向得分截尾 [0.05,0.95]）
匹配协变量加入 2020 年披露状态（政策前披露倾向是关键的混杂因子）。
每个方法输出：平衡性（最大标准化偏差）+ 加权/匹配样本 DID。
"""
import pandas as pd
import numpy as np
import json

WS = r"D:\7 论文集\26暑 巴黎协定\全球气候治理与企业跨区域投资——基于《巴黎协定》的准自然实验 (3)"
df = pd.read_parquet(WS + r"\carbon_panel_2019_2024.parquet")
LAGC = ['L1_' + c for c in ['Size', 'Lev', 'ROA', 'Growth', 'Cash', 'TobinQ', 'Age',
                            'SOE', 'Top1', 'Board', 'Indep', 'Dual']]

def star(t):
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.96 else ('*' if a > 1.645 else ''))

def twfe(d, ycol, xcols, firm='stkcd', tcol='year', weights=None, max_iter=300):
    d = d[[ycol] + xcols + [firm, tcol] + ([weights] if weights else [])].dropna().copy()
    N = len(d); y = d[ycol].to_numpy(float); X = d[xcols].to_numpy(float)
    w = d[weights].to_numpy(float) if weights else np.ones(N)
    sw = np.sqrt(w)
    f = d[firm].to_numpy(); t = d[tcol].to_numpy()
    def demean(M):
        if weights is None:
            for _ in range(max_iter):
                prev = M.copy()
                M = M - pd.DataFrame(M).groupby(f).transform('mean').to_numpy()
                M = M - pd.DataFrame(M).groupby(t).transform('mean').to_numpy()
                if np.abs(M - prev).max() < 1e-10: break
            return M
        f_i = pd.factorize(f)[0]
        t_i = pd.factorize(t)[0]
        def wmean(M, idx):
            W = np.bincount(idx, weights=w)
            out = np.empty_like(M)
            for j in range(M.shape[1]):
                s = np.bincount(idx, weights=w * M[:, j])
                out[:, j] = (s / W)[idx]
            return out
        for _ in range(max_iter):
            prev = M.copy()
            M = M - wmean(M, f_i)
            M = M - wmean(M, t_i)
            if np.abs(M - prev).max() < 1e-10: break
        return M
    yr = demean(y.reshape(-1, 1)).flatten()
    Xr = demean(X)
    Xw = Xr * sw.reshape(-1, 1); yw = yr * sw
    A = Xw.T @ Xr
    beta = np.linalg.solve(A, Xw.T @ yw)
    u = yr - Xr @ beta
    _, fidx = np.unique(f, return_inverse=True)
    k = len(xcols)
    s = np.zeros((fidx.max() + 1, k))
    np.add.at(s, fidx, Xr * (w * u).reshape(-1, 1))
    V = np.linalg.inv(A) @ (s.T @ s) @ np.linalg.inv(A)
    G = s.shape[0]
    V *= (G / (G - 1)) * ((N - 1) / (N - k))
    se = np.sqrt(np.diag(V))
    return dict(N=N, G=G, b=float(beta[0]), se=float(se[0]), t=float(beta[0]/se[0]))

# ---------- 2020 截面与倾向得分 ----------
EB = ['Size', 'Lev', 'ROA', 'Growth', 'Cash', 'TobinQ', 'Age', 'SOE', 'Top1']
cs = df[(df['year'] == 2020) & df[EB].notna().all(axis=1)].copy()
disc20 = df[(df['year'] == 2020)].set_index('stkcd')['Disclose']
cs['Disc20'] = cs['stkcd'].map(disc20).fillna(0)
MVARS = EB + ['Disc20']
tm = cs['Treat'].to_numpy() == 1
X = cs[MVARS].to_numpy(float)
mu, sd = X[~tm].mean(0), X[~tm].std(0); sd[sd == 0] = 1
Xs = (X - mu) / sd

def logit_fit(Xd, y, iters=200):
    X1 = np.column_stack([np.ones(len(Xd)), Xd])
    b = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ b))
        W = np.maximum(p * (1 - p), 1e-9)
        g = X1.T @ (y - p)
        H = (X1 * W.reshape(-1, 1)).T @ X1 + 1e-6 * np.eye(X1.shape[1])
        st = np.linalg.solve(H, g)
        b += st
        if np.abs(st).max() < 1e-9: break
    return b

b = logit_fit(Xs, tm.astype(float))
ps = 1 / (1 + np.exp(-np.column_stack([np.ones(len(Xs)), Xs]) @ b))
cs['ps'] = ps
lp = np.log(ps / (1 - ps))
nt = int(tm.sum()); nc = int((~tm).sum())
t_idx = np.where(tm)[0]; c_idx = np.where(~tm)[0]

def std_bias(mask_c):
    full = np.zeros(len(X), bool)
    full[c_idx] = mask_c
    out = []
    for k in range(len(MVARS)):
        mt = X[tm, k].mean(); mc = X[full, k].mean()
        s = X[:, k].std(ddof=1)
        out.append(abs(mt - mc) / s)
    return max(out)

# ---------- 各匹配方法 → 对照组权重 ----------
methods = {}

# ① 1:1 NN 卡尺 0.05
cal = 0.05 * np.std(lp)
used = np.zeros(nc, int)
ok_t = 0
for i in t_idx:
    d = np.abs(lp[c_idx] - lp[i])
    j = int(d.argmin())
    if d[j] <= cal:
        used[j] += 1; ok_t += 1
w = np.zeros(nc); w[used > 0] = 1
methods['① 1:1NN卡尺0.05'] = ('subset', w, f"匹配处理 {ok_t}/{nt}")

# ② 1:1 NN 无卡尺
used = np.zeros(nc, int)
for i in t_idx:
    d = np.abs(lp[c_idx] - lp[i])
    used[int(d.argmin())] += 1
methods['② 1:1NN无卡尺'] = ('subset', (used > 0).astype(float), f"全部 {nt} 处理匹配")

# ③ 1:4 NN（权重=次数）
used = np.zeros(nc, int)
for i in t_idx:
    d = np.abs(lp[c_idx] - lp[i])
    for j in np.argsort(d)[:4]:
        used[j] += 1
methods['③ 1:4NN'] = ('weight', used.astype(float), f"对照被用 {(used>0).sum()} 家")

# ④ 半径匹配 0.05
cal = 0.05 * np.std(lp)
used = np.zeros(nc, int)
for i in t_idx:
    d = np.abs(lp[c_idx] - lp[i])
    used[(d <= cal)] += 1
methods['④ 半径0.05'] = ('weight', used.astype(float), f"对照被用 {(used>0).sum()} 家")

# ⑤ 核匹配（高斯，带宽=0.05×SD）
h = 0.05 * np.std(lp)
kw = np.zeros(nc)
for i in t_idx:
    kw += np.exp(-0.5 * ((lp[c_idx] - lp[i]) / h) ** 2)
methods['⑤ 核匹配h0.05'] = ('weight', kw, f"全部 {nc} 对照加权")

# ⑥ 马氏距离 1:1
VI = np.linalg.inv(np.cov(Xs[~tm].T))
used = np.zeros(nc, int)
md_c = Xs[~tm]
for i in t_idx:
    diff = md_c - Xs[i]
    d = np.sqrt(np.einsum('ij,jk,ik->i', diff, VI, diff))
    used[int(d.argmin())] += 1
methods['⑥ 马氏1:1'] = ('subset', (used > 0).astype(float), f"对照被用 {(used>0).sum()} 家")

# ⑦ IPW-ATT（截尾 0.05-0.95）
psw = np.clip(ps[~tm], 0.05, 0.95)
ipw = psw / (1 - psw)
methods['⑦ IPW-ATT'] = ('weight', ipw, f"截尾 {(ps[~tm]<0.05).sum()}+{(ps[~tm]>0.95).sum()} 家")

# ---------- 估计 ----------
sub = df[df['year'].between(2020, 2024)].copy()
rows = []
print(f"{'方法':<18}{'形式':<8}{'max|偏差|':>9}{'Disclose DID':>22}{'t':>8}  {'Reduce DID':>20}{'t':>8}")
for name, (form, w_c, note) in methods.items():
    mask_c = w_c > 0
    mb = std_bias(mask_c) if form == 'subset' else np.nan
    # 构造面板权重/子样本
    if form == 'subset':
        firms = set(cs['stkcd'].iloc[t_idx]) | set(cs['stkcd'].iloc[c_idx[mask_c]])
        d = sub[sub['stkcd'].isin(firms)].copy()
        r1 = twfe(d, 'Disclose', ['DID'] + LAGC)
        r2 = twfe(d, 'Reduce', ['DID'] + LAGC)
        n_firm = d['stkcd'].nunique()
    else:
        fw = pd.Series(1.0, index=cs['stkcd'].iloc[t_idx])
        fw = pd.concat([fw, pd.Series(w_c, index=cs['stkcd'].iloc[c_idx])])
        fw = fw[fw > 0]
        c_mean = fw.iloc[nt:].mean() if len(fw) > nt else 1.0
        fw = fw / c_mean  # 对照组均权=1
        # 加权平衡性（仅存活对照）
        keep_c = w_c > 0
        wm = fw.iloc[nt:].to_numpy()
        xw = X[~tm][keep_c] * (wm / wm.sum()).reshape(-1, 1)
        mb = max(abs(X[tm, k].mean() - xw[:, k].sum()) / X[:, k].std(ddof=1) for k in range(len(MVARS)))
        d = sub.copy()
        d['mw'] = d['stkcd'].map(fw)
        d = d[d['mw'].notna()].copy()
        r1 = twfe(d, 'Disclose', ['DID'] + LAGC, weights='mw')
        r2 = twfe(d, 'Reduce', ['DID'] + LAGC, weights='mw')
        n_firm = d['stkcd'].nunique()
    rows.append({'方法': name, '说明': note, '形式': form,
                 'max标准化偏差': None if np.isnan(mb) else round(float(mb), 3),
                 'Disclose_b': round(r1['b'], 4), 'Disclose_se': round(r1['se'], 4),
                 'Disclose_t': round(r1['t'], 3), 'Disclose_sig': star(r1['t']),
                 'Reduce_b': round(r2['b'], 4), 'Reduce_se': round(r2['se'], 4),
                 'Reduce_t': round(r2['t'], 3), 'Reduce_sig': star(r2['t']),
                 '企业数': n_firm})
    mb_s = f"{mb:.3f}" if not np.isnan(mb) else "  —  "
    print(f"{name:<18}{form:<8}{mb_s:>9}{r1['b']:>+12.4f}{star(r1['t']):>5}{r1['t']:>8.3f}  {r2['b']:>+11.4f}{star(r2['t']):>4}{r2['t']:>8.3f}")

with open(WS + r"\matching_suite_results.json", 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
print("\n已保存 matching_suite_results.json")
