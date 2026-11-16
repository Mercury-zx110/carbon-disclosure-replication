# -*- coding: utf-8 -*-
"""
第五步（碳减排研究）：连续处理（剂量反应）设计
Intensity：按官方高耗能/控排行业分类构造 0—4 有序强度得分（透明、可替换）
  4 = D44 电力热力；3 = C31 钢铁冶炼、C32 有色冶炼、B06 煤炭开采、C25 石化炼焦；
  2 = B07 油气开采、C26 化工、C28 化纤、C30 建材；1 = B08—B12 其他采矿、D45 燃气；0 = 其他
检验：① 连续 DID（Intensity×Post）② 剂量梯度（强度等级×Post 分组）③ 替代评分稳健
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
ind = df.loc[df['year'] == 2020, ['stkcd', 'indcode_csrc']]
ind['i3'] = ind['indcode_csrc'].astype(str).str[:3]
ind['Intensity'] = ind['i3'].map(INT).fillna(0)
df = df.merge(ind[['stkcd', 'Intensity']], on='stkcd', how='left')
df['Intensity'] = df['Intensity'].fillna(0)
df['IntPost'] = df['Intensity'] * df['Post']
df['IntPost22'] = df['Intensity'] * df['Post22']
for k in [1, 2, 3, 4]:
    df[f'Tier{k}'] = (df['Intensity'] == k).astype(int)
    df[f'Tier{k}Post'] = df[f'Tier{k}'] * df['Post']
    df[f'Tier{k}Post22'] = df[f'Tier{k}'] * df['Post22']
# 替代评分 A：强度≥3 记 1，其余 0（等价于"核心控排行业"）
df['IntA'] = (df['Intensity'] >= 3).astype(int)
df['IntAPost'] = df['IntA'] * df['Post']
# 替代评分 B：0—4 线性但 D44 与 3 级拉平（去极值）
INTB = {k: (3 if v == 4 else v) for k, v in INT.items()}
df['IntB'] = df['i3'] = None
ind['IntB'] = ind['i3'].map(INTB).fillna(0)
df = df.drop(columns=['IntB']).merge(ind[['stkcd', 'IntB']], on='stkcd', how='left') if 'IntB' in df else df
if 'IntB' not in df:
    df = df.merge(ind[['stkcd', 'IntB']], on='stkcd', how='left')
df['IntBPost'] = df['IntB'] * df['Post']

print("强度得分分布（2020 行业层面）：")
print(ind.groupby(['i3', 'Intensity']).size().rename('公司数').reset_index().sort_values('Intensity', ascending=False).head(18).to_string(index=False))

def star(t):
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.96 else ('*' if a > 1.645 else ''))

def twfe(d, ycol, xcols, firm='stkcd', tcol='year', max_iter=300):
    d = d[[ycol] + xcols + [firm, tcol]].dropna().copy()
    N = len(d); y = d[ycol].to_numpy(float); X = d[xcols].to_numpy(float)
    f = d[firm].to_numpy(); t = d[tcol].to_numpy()
    def demean(M):
        for _ in range(max_iter):
            prev = M.copy()
            M = M - pd.DataFrame(M).groupby(f).transform('mean').to_numpy()
            M = M - pd.DataFrame(M).groupby(t).transform('mean').to_numpy()
            if np.abs(M - prev).max() < 1e-10: break
        return M
    yr = demean(y.reshape(-1, 1)).flatten(); Xr = demean(X)
    XtX = Xr.T @ Xr
    beta = np.linalg.lstsq(Xr, yr, rcond=None)[0] if np.linalg.matrix_rank(XtX) < len(xcols) else np.linalg.solve(XtX, Xr.T @ yr)
    u = yr - Xr @ beta
    _, fidx = np.unique(f, return_inverse=True)
    k = len(xcols)
    s = np.zeros((fidx.max() + 1, k))
    np.add.at(s, fidx, Xr * u.reshape(-1, 1))
    XtXi = np.linalg.pinv(XtX)
    V = XtXi @ (s.T @ s) @ XtXi
    G = s.shape[0]
    V *= (G / (G - 1)) * ((N - 1) / (N - k))
    se = np.sqrt(np.diag(V))
    t_all = beta / se
    r2 = 1 - (u ** 2).sum() / (yr ** 2).sum()
    return dict(N=N, G=G, b=beta.tolist(), se=se.tolist(), t=t_all.tolist(), r2=float(r2))

sub = df[df['year'].between(2020, 2024)].copy()
out = []

def rep(tag, r, i=0):
    print(f"{tag:<46} b={r['b'][i]:+.4f} (se={r['se'][i]:.4f}) t={r['t'][i]:+.3f} {star(r['t'][i])}  N={r['N']}")
    return {'标签': tag, 'b': round(r['b'][i], 4), 'se': round(r['se'][i], 4),
            't': round(r['t'][i], 3), 'sig': star(r['t'][i]), 'N': r['N']}

print("\n===== ① 连续剂量 DID =====")
out.append(rep("Disclose ~ Intensity×Post", twfe(sub, 'Disclose', ['IntPost'] + LAGC)))
out.append(rep("Disclose ~ Intensity×Post22", twfe(sub, 'Disclose', ['IntPost22'] + LAGC)))
out.append(rep("Reduce ~ Intensity×Post", twfe(sub, 'Reduce', ['IntPost'] + LAGC)))
out.append(rep("Reduce ~ Intensity×Post22", twfe(sub, 'Reduce', ['IntPost22'] + LAGC)))

print("\n===== ② 剂量梯度（Tier1—4 × Post 联合估计，基期 2020）=====")
r = twfe(sub, 'Disclose', ['Tier1Post', 'Tier2Post', 'Tier3Post', 'Tier4Post'] + LAGC)
for i, k in enumerate([1, 2, 3, 4]):
    out.append(rep(f"Disclose Tier{k}(强度={k})×Post", r, i))
r2_ = twfe(sub, 'Reduce', ['Tier1Post', 'Tier2Post', 'Tier3Post', 'Tier4Post'] + LAGC)
for i, k in enumerate([1, 2, 3, 4]):
    out.append(rep(f"Reduce Tier{k}(强度={k})×Post", r2_, i))

print("\n===== ③ 替代评分稳健性 =====")
out.append(rep("Disclose ~ 替代A(强度≥3)×Post", twfe(sub, 'Disclose', ['IntAPost'] + LAGC)))
out.append(rep("Disclose ~ 替代B(D44拉平)×Post", twfe(sub, 'Disclose', ['IntBPost'] + LAGC)))
out.append(rep("Reduce ~ 替代A(强度≥3)×Post", twfe(sub, 'Reduce', ['IntAPost'] + LAGC)))
out.append(rep("Reduce ~ 替代B(D44拉平)×Post", twfe(sub, 'Reduce', ['IntBPost'] + LAGC)))

with open(WS + r"\dose_response_results.json", 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n已保存 dose_response_results.json")
