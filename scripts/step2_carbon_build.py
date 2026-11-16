# -*- coding: utf-8 -*-
"""
第二步（碳减排研究）：CNDD-0272 单位统一 + 合并财务面板 + Treat/Post + 快速基准 DID
"""
import pandas as pd
import numpy as np
import json

WS = r"D:\7 论文集\26暑 巴黎协定\全球气候治理与企业跨区域投资——基于《巴黎协定》的准自然实验 (3)"

# ---------- 1. 碳减排数据：单位统一 ----------
carbon, _ = __import__('pyreadstat').read_dta(WS + r"\data_carbon\CNDD-0272+上市公司碳减排情况.dta")
UNIT_MAP = {
    '': 1.0, '吨': 1.0, '吨二氧化碳': 1.0, '吨二氧化碳当量': 1.0, '吨/年': 1.0,
    '万吨': 1e4, '万吨/年': 1e4, '万吨二氧化碳': 1e4, '万吨二氧化碳当量': 1e4,
    '公吨': 1.0, '公吨二氧化碳当量': 1.0, '千克': 1e-3,
}
carbon['unit_factor'] = carbon['ghgemissredunit'].astype(str).str.strip().map(UNIT_MAP)
n_bad_unit = int(carbon['unit_factor'].isna().sum())
carbon = carbon[carbon['unit_factor'].notna()].copy()
carbon['co2_ton'] = carbon['co2emissred'] * carbon['unit_factor']
carbon['stkcd'] = carbon['stkcd'].astype(str).str.zfill(6)
n_neg0 = int((carbon['co2_ton'] < 0).sum())

# 【清洗规则 R1】单位空缺且 |值| ≥ 1e8 吨：判定为"万吨"误标，数值 ÷1e4；
# 修正后仍 ≥1e8 吨（单体企业年减排不可能超过 1 亿吨）则剔除
big_blank = (carbon['ghgemissredunit'].astype(str).str.strip() == '') & (carbon['co2_ton'].abs() >= 1e8)
carbon.loc[big_blank, 'co2_ton'] = carbon.loc[big_blank, 'co2_ton'] / 1e4
n_fixed = int(big_blank.sum())
n_impl = int((carbon['co2_ton'].abs() >= 1e8).sum())
dropped_ids = carbon.loc[carbon['co2_ton'].abs() >= 1e8, ['stkcd', 'shortname', 'year', 'co2_ton']]
carbon = carbon[carbon['co2_ton'].abs() < 1e8].copy()
print(f"[1] 单位无法识别剔除 {n_bad_unit} 条；R1 单位修正 {n_fixed} 条；物理不可行剔除 {n_impl} 条：")
print(dropped_ids.to_string(index=False))
print(f"    负值（排放上升）保留 {n_neg0} 条")

# 同公司-年度多条取绝对值最大者（主记录）
carbon = (carbon.sort_values(['stkcd', 'year', 'co2_ton'])
               .groupby(['stkcd', 'year'], as_index=False).last())

# ---------- 2. 合并财务面板 ----------
panel = pd.read_parquet(WS + r"\cleaned_panel_2019_2024.parquet")
panel['stkcd'] = panel['stkcd'].astype(str).str.zfill(6)
df = panel.merge(carbon[['stkcd', 'year', 'co2_ton', 'ghgemissred', 'cemissred']], on=['stkcd', 'year'], how='left')
df['Disclose'] = df['co2_ton'].notna().astype(float)
df['NegRed'] = (df['co2_ton'] < 0).astype(float)
df['Reduce'] = np.where(df['co2_ton'].notna(), np.log1p(df['co2_ton']), np.nan)
df['Reduce_pos'] = np.where(df['co2_ton'] > 0, np.log(df['co2_ton']), np.nan)

# ---------- 3. 缩尾（披露子样本内 1%/99%） ----------
for c in ['Reduce', 'Reduce_pos']:
    sub = df[c].notna()
    lo, hi = df.loc[sub, c].quantile(0.01), df.loc[sub, c].quantile(0.99)
    df.loc[sub, c] = df.loc[sub, c].clip(lo, hi)

# ---------- 4. Treat / Post ----------
HC = ['B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12',
      'C25', 'C26', 'C28', 'C30', 'C31', 'C32', 'D44', 'D45']
ind20 = (df.loc[df['year'] == 2020, ['stkcd', 'indcode_csrc']]
         .assign(hc=lambda x: x['indcode_csrc'].astype(str).str[:3].isin(HC).astype(int))
         [['stkcd', 'hc']].rename(columns={'hc': 'Treat'}))
df = df.merge(ind20, on='stkcd', how='left')
# 【清洗规则 R2】2020 年无记录的公司（Treat 无法依政策前行业定义，主要为 2021 后新上市）
# 整体剔除，避免被误标为对照组
n_nofirms = df['Treat'].isna().sum()
df = df[df['Treat'].notna()].copy()
df['Treat'] = df['Treat'].astype(int)
print(f"    R2 剔除 2020 年无记录公司观测 {int(n_nofirms)} 条")
df['Post'] = (df['year'] >= 2021).astype(int)
df['DID'] = df['Treat'] * df['Post']
df['Post22'] = (df['year'] >= 2022).astype(int)
df['DID22'] = df['Treat'] * df['Post22']

# 滞后一期控制变量
CONTROLS = ['Size', 'Lev', 'ROA', 'Growth', 'Cash', 'TobinQ', 'Age', 'SOE', 'Top1', 'Board', 'Indep', 'Dual']
df = df.sort_values(['stkcd', 'year'])
for c in CONTROLS:
    df['L1_' + c] = df.groupby('stkcd')[c].shift(1)
LAGC = ['L1_' + c for c in CONTROLS]

df.to_parquet(WS + r"\carbon_panel_2019_2024.parquet", index=False)
n_disclose = int(df['Disclose'].sum())
print(f"[2] 合并完成：{len(df)} 条，披露 {n_disclose} 条（{(df['Disclose'].mean()*100):.1f}%）")
print(f"    处理组公司：{df.groupby('stkcd')['Treat'].first().sum()} / {df['stkcd'].nunique()}")
print(f"    披露×处理组：{int((df['Disclose']*df['Treat']).sum())} 条")

# ---------- 5. 快速基准 DID ----------
def twfe(d, ycol, xcols, firm='stkcd', tcol='year'):
    d = d[[ycol] + xcols + [firm, tcol]].dropna().copy()
    N = len(d); y = d[ycol].to_numpy(float); X = d[xcols].to_numpy(float)
    def demean(M):
        for _ in range(300):
            prev = M.copy()
            M = M - pd.DataFrame(M).groupby(d[firm].to_numpy()).transform('mean').to_numpy()
            M = M - pd.DataFrame(M).groupby(d[tcol].to_numpy()).transform('mean').to_numpy()
            if np.abs(M - prev).max() < 1e-10: break
        return M
    yr = demean(y.reshape(-1, 1)).flatten(); Xr = demean(X)
    XtX = Xr.T @ Xr
    beta = np.linalg.solve(XtX, Xr.T @ yr)
    u = yr - Xr @ beta
    _, fidx = np.unique(d[firm].to_numpy(), return_inverse=True)
    k = len(xcols)
    s = np.zeros((fidx.max() + 1, k))
    np.add.at(s, fidx, Xr * u.reshape(-1, 1))
    V = np.linalg.inv(XtX) @ (s.T @ s) @ np.linalg.inv(XtX)
    G = s.shape[0]
    V *= (G / (G - 1)) * ((N - 1) / (N - k))
    se = np.sqrt(np.diag(V))
    r2 = 1 - (u ** 2).sum() / (yr ** 2).sum()
    return dict(N=N, G=G, b=float(beta[0]), se=float(se[0]), t=float(beta[0]/se[0]), r2=float(r2))

def star(t):
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.96 else ('*' if a > 1.645 else ''))

sub = df[df['year'].between(2020, 2024)].copy()
specs = [
    ('Reduce',    'DID（Post=2021）', ['DID'] + LAGC),
    ('Reduce',    'DID（Post=2022）', ['DID22'] + LAGC),
    ('Reduce_pos','DID（正值口径）',  ['DID'] + LAGC),
    ('Disclose',  '披露LPM（全样本）', ['DID'] + LAGC),
    ('Disclose',  '披露LPM（Post22）', ['DID22'] + LAGC),
]
print("\n===== 快速基准 DID（2020—2024，双向FE+企业聚类SE）=====")
out = []
for y, label, xs in specs:
    r = twfe(sub, y, xs)
    out.append({'y': y, '设定': label, **r, 'sig': star(r['t'])})
    print(f"{y:<11}{label:<16} b={r['b']:+.4f} (se={r['se']:.4f}) t={r['t']:+.3f} {star(r['t'])}  N={r['N']} 企业={r['G']} R2={r['r2']:.4f}")
with open(WS + r"\carbon_baseline_quick.json", 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

# 分组年度均值（事实刻画）
print("\n===== 披露率与减排量分组轨迹 =====")
t = sub.groupby(['year', 'Treat']).apply(
    lambda g: pd.Series({'披露率': g['Disclose'].mean(),
                         '减排量中位数(吨)': g['co2_ton'].median()}),
    include_groups=False).unstack()
print(t.round(3))
