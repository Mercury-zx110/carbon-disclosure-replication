# -*- coding: utf-8 -*-
"""
第六步（碳减排研究）：剂量反应设计的完整识别链
  1) 伪时点安慰剂（虚构政策在 2022，"政策前"=2020—2021 → 系数应不显著）
  2) 随机化强度分配安慰剂（300 次）
  3) 动态剂量效应（Intensity×Year，基期 2020）
  4) 省份×年份联合固定效应（吸收区域性政策冲击）
  5) 处理组内再验证：仅高碳行业子样本的强度梯度（剔除行业间构成）
"""
import pandas as pd
import numpy as np
import json

WS = r"D:\7 论文集\26暑 巴黎协定\全球气候治理与企业跨区域投资——基于《巴黎协定》的准自然实验 (3)"
df = pd.read_parquet(WS + r"\carbon_panel_2019_2024.parquet")
LAGC = ['L1_' + c for c in ['Size', 'Lev', 'ROA', 'Growth', 'Cash', 'TobinQ', 'Age',
                            'SOE', 'Top1', 'Board', 'Indep', 'Dual']]

INT = {'D44': 4, 'D45': 1, 'C31': 3, 'C32': 3, 'B06': 3, 'C25': 3,
       'B07': 2, 'C26': 2, 'C28': 2, 'C30': 2,
       'B08': 1, 'B09': 1, 'B10': 1, 'B11': 1, 'B12': 1}
ind = df.loc[df['year'] == 2020, ['stkcd', 'indcode_csrc']].copy()
ind['i3'] = ind['indcode_csrc'].astype(str).str[:3]
ind['Intensity'] = ind['i3'].map(INT).fillna(0)
df = df.merge(ind[['stkcd', 'Intensity']], on='stkcd', how='left')
df['Intensity'] = df['Intensity'].fillna(0)
df['Post'] = (df['year'] >= 2021).astype(int)
df['Post22'] = (df['year'] >= 2022).astype(int)
df['IntPost'] = df['Intensity'] * df['Post']
df['IntPost22'] = df['Intensity'] * df['Post22']
# 伪时点：虚构政策 2022 年生效（Post_fake=2022+），在 2020—2021 "政策前"窗口内应无效应
df['PostFake'] = (df['year'] >= 2022).astype(int)
df['IntPostFake'] = df['Intensity'] * df['PostFake']

def star(t):
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.96 else ('*' if a > 1.645 else ''))

def twfe(d, ycol, xcols, firm='stkcd', tcol='year', extra_fe=None, max_iter=300):
    cols = [ycol] + xcols + [firm, tcol] + ([extra_fe] if extra_fe else [])
    d = d[cols].dropna().copy()
    N = len(d); y = d[ycol].to_numpy(float); X = d[xcols].to_numpy(float)
    f = d[firm].to_numpy(); t = d[tcol].to_numpy()
    fe2 = d[extra_fe].to_numpy() if extra_fe else None
    def demean(M):
        for _ in range(max_iter):
            prev = M.copy()
            M = M - pd.DataFrame(M).groupby(f).transform('mean').to_numpy()
            M = M - pd.DataFrame(M).groupby(t).transform('mean').to_numpy()
            if fe2 is not None:
                M = M - pd.DataFrame(M).groupby(fe2).transform('mean').to_numpy()
            if np.abs(M - prev).max() < 1e-10: break
        return M
    yr = demean(y.reshape(-1, 1)).flatten(); Xr = demean(X)
    XtX = Xr.T @ Xr
    beta = np.linalg.lstsq(Xr, yr, rcond=None)[0]
    u = yr - Xr @ beta
    _, fidx = np.unique(f, return_inverse=True)
    k = len(xcols)
    s = np.zeros((fidx.max() + 1, k))
    np.add.at(s, fidx, Xr * u.reshape(-1, 1))
    XtXi = np.linalg.pinv(XtX)
    V = XtXi @ (s.T @ s) @ XtXi
    G = s.shape[0]
    V *= (G / (G - 1)) * ((N - 1) / max(N - k, 1))
    se = np.sqrt(np.diag(V))
    t_all = beta / se
    r2 = 1 - (u ** 2).sum() / (yr ** 2).sum()
    return dict(N=N, G=G, b=beta.tolist(), se=se.tolist(), t=t_all.tolist(), r2=float(r2))

sub = df[df['year'].between(2020, 2024)].copy()
out = []

def rep(tag, r, i=0):
    b = r['b'][i]; se = r['se'][i]; t = r['t'][i]
    t_s = f"{t:+.3f}" if np.isfinite(t) else "nan"
    print(f"{tag:<46} b={b:+.4f} (se={se:.4f}) t={t_s} {star(t) if np.isfinite(t) else ''}  N={r['N']}")
    return {'标签': tag, 'b': round(float(b), 4), 'se': None if not np.isfinite(se) else round(float(se), 4),
            't': None if not np.isfinite(t) else round(float(t), 3),
            'sig': star(t) if np.isfinite(t) else 'collinear', 'N': r['N']}

print("===== 1) 伪时点安慰剂 =====")
pre = sub[sub['year'].isin([2020, 2021])].copy()
out.append(rep("伪时点 IntPostFake（2020—21窗口）", twfe(pre, 'Disclose', ['IntPostFake'] + LAGC)))

print("\n===== 2) 随机化强度安慰剂（300 次）=====")
rng = np.random.default_rng(20261024)
i3_map = sub.groupby('stkcd')['Intensity'].first()
levels = i3_map[i3_map > 0].to_numpy()
fake_ts = []
fmap = sub['stkcd'].to_numpy()
for _ in range(300):
    fake_lev = dict(zip(i3_map.index, rng.permutation(i3_map.to_numpy())))
    fake_i = sub['stkcd'].map(fake_lev).to_numpy() * sub['Post'].to_numpy()
    d = sub.copy(); d['fakeX'] = fake_i
    r = twfe(d, 'Disclose', ['fakeX'] + LAGC, max_iter=80)
    fake_ts.append(r['t'][0])
fake_ts = np.array(fake_ts)
real = twfe(sub, 'Disclose', ['IntPost'] + LAGC)
real_t = real['t'][0]
p_fake = float((np.abs(fake_ts) >= abs(real_t)).mean())
print(f"真实 t={real_t:.3f}；随机化 300 次 |t|：P95={np.percentile(np.abs(fake_ts),95):.3f} "
      f"max={np.abs(fake_ts).max():.3f}，经验 p={p_fake:.4f}")
out.append({'标签': '随机化强度安慰剂', 'b': None, 'se': None, 't': round(float(real_t), 3),
            'sig': f'经验p={p_fake:.4f}', 'N': 300})

print("\n===== 3) 动态剂量效应（Intensity×Year，基期 2020）=====")
d = sub.copy()
for yy in [2021, 2022, 2023, 2024]:
    d[f'IX{yy}'] = d['Intensity'] * (d['year'] == yy).astype(int)
r = twfe(d, 'Disclose', ['IX2021', 'IX2022', 'IX2023', 'IX2024'] + LAGC)
for i, yy in enumerate([2021, 2022, 2023, 2024]):
    out.append(rep(f"动态剂量 Disclose Intensity×{yy}", r, i))

print("\n===== 4) 省份×年份联合固定效应 =====")
out.append(rep("Disclose IntPost + 省×年FE", twfe(sub, 'Disclose', ['IntPost'] + LAGC, extra_fe='province')))
out.append(rep("Disclose IntPost22 + 省×年FE", twfe(sub, 'Disclose', ['IntPost22'] + LAGC, extra_fe='province')))

print("\n===== 5) 高碳行业内部强度梯度（剔除行业间构成）=====")
hc = sub[sub['Intensity'] > 0].copy()
hc['IntPost_h'] = hc['Intensity'] * hc['Post']
out.append(rep("高碳子样本内 Disclose Intensity×Post", twfe(hc, 'Disclose', ['IntPost_h'] + LAGC)))

with open(WS + r"\dose_identification_results.json", 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n已保存 dose_identification_results.json")
