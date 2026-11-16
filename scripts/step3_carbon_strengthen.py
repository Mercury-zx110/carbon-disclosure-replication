# -*- coding: utf-8 -*-
"""
第三步（碳减排研究）：识别加固
  1) 持续披露子样本（剔除"谁来披露"的构成效应）
  2) 熵平衡（2020 年截面匹配，加权 DID）
  3) 两部分模型正式估计（式2 披露 LPM 全样本 + 式3 减排量披露子样本）
  4) 安慰剂：随机化处理组 500 次
  5) 动态效应：2021—2024 逐年系数（基期 2020）
"""
import pandas as pd
import numpy as np
import json

WS = r"D:\7 论文集\26暑 巴黎协定\全球气候治理与企业跨区域投资——基于《巴黎协定》的准自然实验 (3)"
df = pd.read_parquet(WS + r"\carbon_panel_2019_2024.parquet")
LAGC = ['L1_' + c for c in ['Size', 'Lev', 'ROA', 'Growth', 'Cash', 'TobinQ', 'Age',
                            'SOE', 'Top1', 'Board', 'Indep', 'Dual']]

# ---------- 通用工具 ----------
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
        for _ in range(max_iter):
            prev = M.copy()
            for idx in (f, t):
                # 加权组均值
                means = {}
                for j in range(M.shape[1]):
                    s = pd.Series(M[:, j])
                    gm = s.groupby(idx).apply(lambda x: np.average(x, weights=w[x.index]))
                    means[j] = gm
                mm = np.column_stack([means[j].loc[idx].to_numpy() for j in range(M.shape[1])])
                M = M - mm
            if np.abs(M - prev).max() < 1e-10: break
        return M
    yr = demean(y.reshape(-1, 1)).flatten()
    Xr = demean(X)
    Xw = Xr * sw.reshape(-1, 1); yw = yr * sw
    XtX = Xw.T @ Xr                      # WLS: (X'WX) b = X'Wy
    beta = np.linalg.solve(Xw.T @ Xr, Xw.T @ yw)
    u = yr - Xr @ beta
    _, fidx = np.unique(f, return_inverse=True)
    k = len(xcols)
    s = np.zeros((fidx.max() + 1, k))
    np.add.at(s, fidx, Xr * (w * u).reshape(-1, 1))
    V = np.linalg.inv(Xw.T @ Xr) @ (s.T @ s) @ np.linalg.inv(Xw.T @ Xr)
    G = s.shape[0]
    V *= (G / (G - 1)) * ((N - 1) / (N - k))
    se = np.sqrt(np.diag(V))
    r2 = 1 - (w * u ** 2).sum() / (w * yr ** 2).sum()
    t_all = beta / se
    return dict(N=N, G=G, b=float(beta[0]), se=float(se[0]), t=float(beta[0]/se[0]), r2=float(r2),
                b_all=beta.tolist(), se_all=se.tolist(), t_all=t_all.tolist())

def report(tag, r):
    print(f"{tag:<44} b={r['b']:+.4f} (se={r['se']:.4f}) t={r['t']:+.3f} {star(r['t'])}  N={r['N']} 企业={r['G']}")
    return {'标签': tag, **r, 'sig': star(r['t'])}

sub = df[df['year'].between(2020, 2024)].copy()
out = []

# ---------- 1) 持续披露子样本 ----------
pre_disclose = (sub[sub['year'].isin([2020, 2021])]
                .groupby('stkcd')['Disclose'].min().rename('pre_disc'))
sd = sub.merge(pre_disclose, on='stkcd')
pers = sd[sd['pre_disc'] == 1].copy()
print(f"[1] 持续披露子样本（2020—2021 均披露）：{pers['stkcd'].nunique()} 家企业, {len(pers)} 条")
for y in ['Reduce', 'Reduce_pos']:
    for did in ['DID', 'DID22']:
        out.append(report(f"持续披露子样本 {y} {did}", twfe(pers, y, [did] + LAGC)))

# ---------- 2) 熵平衡（2020 年截面，对照组加权匹配处理组） ----------
EB_VARS = ['Size', 'Lev', 'ROA', 'Growth', 'Cash', 'TobinQ', 'Age', 'SOE', 'Top1']
cs = df[(df['year'] == 2020) & df[EB_VARS].notna().all(axis=1)].copy()
def logit_fit(X, y, iters=100):
    X1 = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ b))
        W = p * (1 - p)
        g = X1.T @ (y - p)
        H = (X1 * W.reshape(-1, 1)).T @ X1 + 1e-6 * np.eye(X1.shape[1])
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8: break
    return b

cs = df[(df['year'] == 2020) & df[EB_VARS].notna().all(axis=1)].copy()
tm = cs['Treat'].to_numpy() == 1
Xraw = cs[EB_VARS].to_numpy(float)
mu, sd = Xraw[~tm].mean(0), Xraw[~tm].std(0); sd[sd == 0] = 1.0
Xs = (Xraw - mu) / sd
b = logit_fit(Xs, tm.astype(float))
ps = 1 / (1 + np.exp(-np.column_stack([np.ones(len(Xs)), Xs]) @ b))
cs['ps'] = ps
lo_ps, hi_ps = ps[tm].min(), ps[tm].max()
keep = (ps >= lo_ps) & (ps <= hi_ps)
cs_cs = cs[keep].copy(); ps_k = ps[keep]; tm_k = tm[keep]
logit_ps = np.log(ps_k / (1 - ps_k))
caliper = 0.05 * np.std(logit_ps)
matched_pairs = []
for i in np.where(tm_k)[0]:
    d = np.abs(logit_ps - logit_ps[i])
    d[tm_k] = np.inf
    j = int(d.argmin())
    if d[j] <= caliper:
        matched_pairs.append((i, j))
treat_idx = sorted(set(pp[0] for pp in matched_pairs))
ctrl_idx = sorted(set(pp[1] for pp in matched_pairs))
matched_firms = set(cs_cs['stkcd'].iloc[treat_idx + ctrl_idx])
print(f"\n[2] PSM：公共支撑内处理 {tm_k.sum()} 家，匹配成功 {len(treat_idx)} 家（卡尺 {caliper:.4f}），唯一公司 {len(matched_firms)} 家")
bal = []
for k, v in enumerate(EB_VARS):
    mt = Xraw[keep][tm_k, k].mean()
    mc = Xraw[keep][~tm_k, k].mean()
    mm = Xraw[keep][ctrl_idx, k].mean()
    st = np.std(Xraw[keep][:, k], ddof=1)
    bal.append({'变量': v, '处理组均值': round(mt, 3), '原始对照均值': round(mc, 3),
                '匹配对照均值': round(mm, 3), '匹配前标准化偏差': round((mt - mc) / st, 3),
                '匹配后标准化偏差': round((mt - mm) / st, 3)})
bal_df = pd.DataFrame(bal)
print(bal_df.to_string(index=False))
subm = sub[sub['stkcd'].isin(matched_firms)].copy()
out.append(report("PSM 匹配样本 Disclose LPM DID", twfe(subm, 'Disclose', ['DID'] + LAGC)))
out.append(report("PSM 匹配样本 Disclose LPM DID22", twfe(subm, 'Disclose', ['DID22'] + LAGC)))
out.append(report("PSM 匹配样本 Reduce DID", twfe(subm, 'Reduce', ['DID'] + LAGC)))

# ---------- 3) 两部分模型正式表 ----------
out.append(report("式(2) 披露 LPM 全样本 DID [基准]", twfe(sub, 'Disclose', ['DID'] + LAGC)))
out.append(report("式(2) 披露 LPM 全样本 DID22", twfe(sub, 'Disclose', ['DID22'] + LAGC)))
out.append(report("式(3) 减排量（披露子样本）DID", twfe(sub, 'Reduce', ['DID'] + LAGC)))

# ---------- 4) 安慰剂：随机化处理组 500 次（Disclose LPM） ----------
rng = np.random.default_rng(20261023)
firms = sub['stkcd'].unique()
fmap = sub['stkcd'].to_numpy()
n_treat = int(sub.groupby('stkcd')['Treat'].first().sum())
placebo_ts = []
for i in range(300):
    fake_firms = set(rng.choice(firms, size=n_treat, replace=False))
    fake_t = np.array([1 if f in fake_firms else 0 for f in fmap]) * sub['Post'].to_numpy()
    d = sub.copy(); d['fakeDID'] = fake_t
    try:
        r = twfe(d, 'Disclose', ['fakeDID'] + LAGC, max_iter=80)
        placebo_ts.append(r['t'])
    except np.linalg.LinAlgError:
        continue
placebo_ts = np.array(placebo_ts)
real_t = [r['t'] for r in out if r['标签'].startswith('式(2) 披露 LPM 全样本 DID [基准]')][0]
p_placebo = float((np.abs(placebo_ts) >= abs(real_t)).mean())
print(f"\n[4] 安慰剂：真实 t={real_t:.3f}，随机化 300 次 |t| 分布 "
      f"P5={np.percentile(np.abs(placebo_ts),5):.3f} P50={np.percentile(np.abs(placebo_ts),50):.3f} P95={np.percentile(np.abs(placebo_ts),95):.3f}"
      f" max={np.abs(placebo_ts).max():.3f}，经验 p={p_placebo:.4f}")
out.append({'标签': '安慰剂检验', 'N': 300, 'G': int(n_treat), 'b': None, 'se': None,
            't': real_t, 'r2': None, 'sig': f'经验p={p_placebo:.4f}'})

# ---------- 5) 动态效应（基期 2020） ----------
d = sub.copy()
for yy in [2021, 2022, 2023, 2024]:
    d[f'T{yy}'] = d['Treat'] * (d['year'] == yy).astype(int)
dyn = twfe(d, 'Disclose', ['T2021', 'T2022', 'T2023', 'T2024'] + LAGC)
for i, yy in enumerate([2021, 2022, 2023, 2024]):
    out.append({'标签': f'动态效应 Disclose {yy}年', 'N': dyn['N'], 'G': dyn['G'],
                'b': float(dyn['b_all'][i]), 'se': float(dyn['se_all'][i]), 't': float(dyn['t_all'][i]),
                'r2': dyn['r2'], 'sig': star(dyn['t_all'][i])})
print("\n[5] 动态效应（Disclose，基期 2020）：")
for i, yy in enumerate([2021, 2022, 2023, 2024]):
    print(f"  {yy}: b={dyn['b_all'][i]:+.4f} (se={dyn['se_all'][i]:.4f}) t={dyn['t_all'][i]:+.3f} {star(dyn['t_all'][i])}")
dyn2 = twfe(d, 'Reduce', ['T2021', 'T2022', 'T2023', 'T2024'] + LAGC)
print("动态效应（Reduce，披露者，基期 2020）：")
for i, yy in enumerate([2021, 2022, 2023, 2024]):
    print(f"  {yy}: b={dyn2['b_all'][i]:+.4f} (se={dyn2['se_all'][i]:.4f}) t={dyn2['t_all'][i]:+.3f} {star(dyn2['t_all'][i])}")
    out.append({'标签': f'动态效应 Reduce {yy}年', 'N': dyn2['N'], 'G': dyn2['G'],
                'b': float(dyn2['b_all'][i]), 'se': float(dyn2['se_all'][i]), 't': float(dyn2['t_all'][i]),
                'r2': dyn2['r2'], 'sig': star(dyn2['t_all'][i])})

with open(WS + r"\carbon_strengthen_results.json", 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n已保存 carbon_strengthen_results.json")
