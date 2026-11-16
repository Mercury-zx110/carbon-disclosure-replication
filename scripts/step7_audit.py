# -*- coding: utf-8 -*-
"""
第七步（碳减排研究）：生成前审计
A1. Disclose 定义偏窄：仅用 co2emissred 非缺失定义披露，遗漏了仅披露温室气体/碳减排的记录 → 改用三变量任一非缺失
A2. Reduce 的负值处理：log1p 对 x<=-1 产生 NaN（负值披露被静默剔除）→ asinh 变换可同时容纳正负值，作稳健口径
A3. 标准误聚类层级：处理变量定义在行业（3 位码）层面，仅企业聚类可能低估 SE → 补行业层面与省份层面聚类
A4. R1 单位纠错阈值（1e8）敏感性 → 1e7 阈值重算
"""
import pandas as pd
import numpy as np
import pyreadstat
import json

WS = r"D:\7 论文集\26暑 巴黎协定\全球气候治理与企业跨区域投资——基于《巴黎协定》的准自然实验 (3)"
df = pd.read_parquet(WS + r"\carbon_panel_2019_2024.parquet")
LAGC = ['L1_' + c for c in ['Size', 'Lev', 'ROA', 'Growth', 'Cash', 'TobinQ', 'Age',
                            'SOE', 'Top1', 'Board', 'Indep', 'Dual']]

# ---------- A1：三变量合并的披露定义 ----------
carbon, _ = pyreadstat.read_dta(WS + r"\data_carbon\CNDD-0272+上市公司碳减排情况.dta")
any_disc = carbon[['co2emissred', 'ghgemissred', 'cemissred']].notna().any(axis=1)
only_other = any_disc & carbon['co2emissred'].isna()
print(f"[A1] 仅披露温室气体/碳减排而无 CO₂ 减排列的记录：{int(only_other.sum())} 条"
      f"（占总记录 {any_disc.mean()*100:.1f}% 的披露记录）")
ad = carbon[any_disc][['stkcd', 'year']].copy()
ad['stkcd'] = ad['stkcd'].astype(str).str.zfill(6)
ad['Disclose_any'] = 1.0
df = df.merge(ad.drop_duplicates(['stkcd', 'year']), on=['stkcd', 'year'], how='left')
df['Disclose_any'] = df['Disclose_any'].fillna(0)

# ---------- A2：asinh 口径 ----------
df['Reduce_asinh'] = np.arcsinh(df['co2_ton'])
sub_a = df['Reduce_asinh'].notna()
lo, hi = df.loc[sub_a, 'Reduce_asinh'].quantile(0.01), df.loc[sub_a, 'Reduce_asinh'].quantile(0.99)
df.loc[sub_a, 'Reduce_asinh'] = df.loc[sub_a, 'Reduce_asinh'].clip(lo, hi)
n_neg = int((df['co2_ton'] < 0).sum())
print(f"[A2] 负值记录 {n_neg} 条；asinh 口径下全部保留（log1p 口径下这些记录因 1+x<=0 被剔除）")

# ---------- 强度得分 ----------
INT = {'D44': 4, 'D45': 1, 'C31': 3, 'C32': 3, 'B06': 3, 'C25': 3,
       'B07': 2, 'C26': 2, 'C28': 2, 'C30': 2,
       'B08': 1, 'B09': 1, 'B10': 1, 'B11': 1, 'B12': 1}
ind = df.loc[df['year'] == 2020, ['stkcd', 'indcode_csrc']].copy()
ind['i3'] = ind['indcode_csrc'].astype(str).str[:3]
ind['Intensity'] = ind['i3'].map(INT).fillna(0)
df = df.merge(ind[['stkcd', 'Intensity', 'i3']], on='stkcd', how='left')
df['Intensity'] = df['Intensity'].fillna(0)
df['Post'] = (df['year'] >= 2021).astype(int)
df['Post22'] = (df['year'] >= 2022).astype(int)
df['IntPost'] = df['Intensity'] * df['Post']
df['IntPost22'] = df['Intensity'] * df['Post22']
df['IntA'] = (df['Intensity'] >= 3).astype(int)
df['IntAPost'] = df['IntA'] * df['Post']

def star(t):
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.96 else ('*' if a > 1.645 else ''))

def twfe(d, ycol, xcols, firm='stkcd', tcol='year', cluster=None, max_iter=300):
    cols = [ycol] + xcols + [firm, tcol] + ([cluster] if cluster else [])
    d = d[cols].dropna().copy()
    N = len(d); y = d[ycol].to_numpy(float); X = d[xcols].to_numpy(float)
    f = d[firm].to_numpy(); t = d[tcol].to_numpy()
    cl = d[cluster].to_numpy() if cluster else f
    def demean(M):
        for _ in range(max_iter):
            prev = M.copy()
            M = M - pd.DataFrame(M).groupby(f).transform('mean').to_numpy()
            M = M - pd.DataFrame(M).groupby(t).transform('mean').to_numpy()
            if np.abs(M - prev).max() < 1e-10: break
        return M
    yr = demean(y.reshape(-1, 1)).flatten(); Xr = demean(X)
    XtX = Xr.T @ Xr
    beta = np.linalg.lstsq(Xr, yr, rcond=None)[0]
    u = yr - Xr @ beta
    _, cidx = np.unique(cl, return_inverse=True)
    k = len(xcols)
    s = np.zeros((cidx.max() + 1, k))
    np.add.at(s, cidx, Xr * u.reshape(-1, 1))
    XtXi = np.linalg.pinv(XtX)
    V = XtXi @ (s.T @ s) @ XtXi
    G = s.shape[0]
    V *= (G / (G - 1)) * ((N - 1) / max(N - k, 1))
    se = np.sqrt(np.diag(V))
    t_all = beta / se
    return dict(N=N, G=int(G), b=beta.tolist(), se=se.tolist(), t=t_all.tolist())

sub = df[df['year'].between(2020, 2024)].copy()
out = []
def rep(tag, r, i=0):
    b, se, t = r['b'][i], r['se'][i], r['t'][i]
    print(f"{tag:<52} b={b:+.4f} (se={se:.4f}) t={t:+.3f} {star(t)}  N={r['N']} 聚类数={r['G']}")
    return {'标签': tag, 'b': round(float(b), 4), 'se': round(float(se), 4),
            't': round(float(t), 3), 'sig': star(t), 'N': r['N']}

print("\n===== A1：Disclose_any 口径 =====")
out.append(rep("Disclose_any IntPost（企业聚类）", twfe(sub, 'Disclose_any', ['IntPost'] + LAGC)))
out.append(rep("Disclose_any IntPost22（企业聚类）", twfe(sub, 'Disclose_any', ['IntPost22'] + LAGC)))
out.append(rep("Disclose_any IntA(≥3)Post", twfe(sub, 'Disclose_any', ['IntAPost'] + LAGC)))

print("\n===== A2：asinh 口径 =====")
out.append(rep("Reduce_asinh IntPost", twfe(sub, 'Reduce_asinh', ['IntPost'] + LAGC)))
out.append(rep("Reduce_asinh IntPost22", twfe(sub, 'Reduce_asinh', ['IntPost22'] + LAGC)))

print("\n===== A3：聚类层级 =====")
out.append(rep("Disclose IntPost（行业3位聚类）", twfe(sub, 'Disclose', ['IntPost'] + LAGC, cluster='i3')))
out.append(rep("Disclose IntPost22（行业3位聚类）", twfe(sub, 'Disclose', ['IntPost22'] + LAGC, cluster='i3')))
out.append(rep("Disclose IntPost（省份聚类）", twfe(sub, 'Disclose', ['IntPost'] + LAGC, cluster='province')))

# ---------- A4：R1 阈值 1e7 重算 ----------
UNIT_MAP = {'': 1.0, '吨': 1.0, '吨二氧化碳': 1.0, '吨二氧化碳当量': 1.0, '吨/年': 1.0,
            '万吨': 1e4, '万吨/年': 1e4, '万吨二氧化碳': 1e4, '万吨二氧化碳当量': 1e4,
            '公吨': 1.0, '公吨二氧化碳当量': 1.0, '千克': 1e-3}
for TH in [1e7]:
    c2 = carbon.copy()
    c2['f'] = c2['ghgemissredunit'].astype(str).str.strip().map(UNIT_MAP)
    c2 = c2[c2['f'].notna()].copy()
    c2['co2_ton'] = c2['co2emissredred'] if False else c2['co2emissred'] * c2['f']
    m = (c2['ghgemissredunit'].astype(str).str.strip() == '') & (c2['co2_ton'].abs() >= TH)
    c2.loc[m, 'co2_ton'] /= 1e4
    c2 = c2[c2['co2_ton'].abs() < TH]
    c2['stkcd'] = c2['stkcd'].astype(str).str.zfill(6)
    c2 = c2.groupby(['stkcd', 'year'], as_index=False).last()
    panel = pd.read_parquet(WS + r"\cleaned_panel_2019_2024.parquet")
    panel['stkcd'] = panel['stkcd'].astype(str).str.zfill(6)
    d2 = panel.merge(c2[['stkcd', 'year', 'co2_ton']], on=['stkcd', 'year'], how='left')
    d2['Disclose'] = d2['co2_ton'].notna().astype(float)
    d2['Reduce'] = np.where(d2['co2_ton'].notna(), np.log1p(d2['co2_ton']), np.nan)
    ok = d2['Reduce'].notna()
    lo2, hi2 = d2.loc[ok, 'Reduce'].quantile(0.01), d2.loc[ok, 'Reduce'].quantile(0.99)
    d2.loc[ok, 'Reduce'] = d2.loc[ok, 'Reduce'].clip(lo2, hi2)
    ind2 = d2.loc[d2['year'] == 2020, ['stkcd', 'indcode_csrc']].copy()
    ind2['Intensity'] = ind2['indcode_csrc'].astype(str).str[:3].map(INT).fillna(0)
    d2 = d2.merge(ind2[['stkcd', 'Intensity']], on='stkcd', how='left')
    d2 = d2[d2['Intensity'].notna()].copy()
    d2['Intensity'] = d2['Intensity'].astype(int)
    d2['Post'] = (d2['year'] >= 2021).astype(int)
    d2['Post22'] = (d2['year'] >= 2022).astype(int)
    d2['IntPost'] = d2['Intensity'] * d2['Post']
    d2['IntPost22'] = d2['Intensity'] * d2['Post22']
    d2 = d2.sort_values(['stkcd', 'year'])
    for c in ['Size', 'Lev', 'ROA', 'Growth', 'Cash', 'TobinQ', 'Age', 'SOE', 'Top1', 'Board', 'Indep', 'Dual']:
        d2['L1_' + c] = d2.groupby('stkcd')[c].shift(1)
    s2 = d2[d2['year'].between(2020, 2024)]
    print(f"\n===== A4：R1 阈值 {TH:.0e}（披露 {int(s2['Disclose'].sum())} 条）=====")
    out.append(rep(f"Disclose IntPost（阈值{TH:.0e}）", twfe(s2, 'Disclose', ['IntPost'] + LAGC)))
    out.append(rep(f"Disclose IntPost22（阈值{TH:.0e}）", twfe(s2, 'Disclose', ['IntPost22'] + LAGC)))
    out.append(rep(f"Reduce IntPost（阈值{TH:.0e}）", twfe(s2, 'Reduce', ['IntPost'] + LAGC)))

with open(WS + r"\audit_results.json", 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n已保存 audit_results.json")
