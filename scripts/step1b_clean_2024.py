# -*- coding: utf-8 -*-
"""
第一步（碳减排研究）：将 CNDD-0006 财务面板延展至 2024 年
清洗规则与 step1_clean.py 完全一致（合并报表、剔金融/ST/状态异常/主要缺失/资不抵债、缩尾）
窗口取 2019—2024（2019 年用于构造 2020 年滞后一期控制变量）
输出：cleaned_panel_2019_2024.parquet + clean_log_2024.json
"""
import pandas as pd
import numpy as np
import json

WS = r"D:\7 论文集\26暑 巴黎协定\全球气候治理与企业跨区域投资——基于《巴黎协定》的准自然实验 (3)"

USECOLS = [
    'Stkcd', 'ShortName', 'Accper',
    'IndustryName', 'IndustryCode', 'IndustryNameC', 'IndustryCodeC',
    'PROVINCECODE', 'PROVINCE', 'CITYCODE', 'CITY',
    'LISTINGDATE', 'LISTINGSTATE', 'Typrep', 'EstablishDate', 'DeclareDate',
    'A001000000', 'A002000000', 'A001101000', 'A003000000', 'A003100000', 'A003101000',
    'B001101000', 'B002000000', 'B002000101', 'C001000000',
    'F011201A', 'F050202B', 'F050502B', 'F081602C', 'F030201A', 'F100901A', 'F100801A',
    'PropertyRightsNature', 'ContrshrProportion', 'Boardsize',
    'IndDirectorRatio', 'ConcurrentPosition', 'SharesBalance',
]

log = []
def rec(step, df):
    log.append((step, len(df)))
    print(f"[{len(log):02d}] {step}: {len(df)} 条", flush=True)
    return df

import pyreadstat
DTA = r"D:\7 论文集\26暑 巴黎协定\CNDD-0006 上市公司最全变量合集（26.1.22更新）\CNDD-0006 上市公司最全变量合集.dta"
df, _ = pyreadstat.read_dta(DTA, usecols=USECOLS)
print("dta 读取完成，总行数:", len(df), flush=True)

df['Accper'] = df['Accper'].astype(str).str.strip()
df['year'] = pd.to_numeric(df['Accper'].str[:4], errors='coerce')
df = df[df['year'].between(2019, 2024)]
rec("原始读入(2019-2024)", df)

df = df[df['Typrep'].astype(str).str.strip().eq('A')]
df = df[df['Accper'].str[5:10].eq('12-31')]
rec("①合并报表+年报", df)

df['DeclareDate'] = df['DeclareDate'].astype(str).str.strip()
before = len(df)
df = (df.sort_values(['Stkcd', 'DeclareDate'])
        .drop_duplicates(subset=['Stkcd', 'year'], keep='last'))
rec(f"①b 公司-年度去重(剔除{before-len(df)}条)", df)

num_cols = ['A001000000', 'A002000000', 'A001101000', 'A003000000', 'A003100000',
            'B001101000', 'B002000000', 'B002000101', 'C001000000',
            'F011201A', 'F050202B', 'F050502B', 'F081602C', 'F030201A',
            'F100901A', 'F100801A', 'ContrshrProportion', 'Boardsize',
            'IndDirectorRatio', 'SharesBalance']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['PropertyRightsNature'] = pd.to_numeric(df['PropertyRightsNature'], errors='coerce')
df['ConcurrentPosition'] = pd.to_numeric(df['ConcurrentPosition'], errors='coerce')

ind = df['IndustryCodeC'].fillna(df['IndustryCode']).astype(str).str.strip().str.upper()
df['IndustryCode_used'] = ind
n_fin = ind.str.startswith('J').sum()
df = df[~ind.str.startswith('J')]
rec(f"②剔除金融业(剔除{n_fin}条)", df)

name = df['ShortName'].astype(str).str.upper().str.strip()
is_st = name.str.contains('ST') | name.str.startswith('PT')
df = df[~is_st]
rec(f"③剔除ST/PT(剔除{int(is_st.sum())}条)", df)

ls = df['LISTINGSTATE'].astype(str).str.strip()
bad = ~ls.isin(['正常上市', '', 'nan', 'None'])
df = df[~bad]
rec(f"④剔除上市状态异常(剔除{int(bad.sum())}条)", df)

before = len(df)
core = ['A001000000', 'B001101000', 'B002000000', 'F011201A']
df = df.dropna(subset=core)
rec(f"⑤剔除主要变量缺失(剔除{before-len(df)}条)", df)

before = len(df)
df = df[(df['F011201A'] > 0) & (df['F011201A'] <= 1)]
rec(f"⑥剔除资不抵债(剔除{before-len(df)}条)", df)

df['Size'] = np.log(df['A001000000'])
df['Lev'] = df['F011201A']
df['ROA'] = df['F050202B']
df['ROE'] = df['F050502B']
df['Growth'] = df['F081602C']
df['Cash'] = df['A001101000'] / df['A001000000']
df['CashRatio'] = df['F030201A']
df['TobinQ'] = df['F100901A']
df['lnMktCap'] = np.log(df['F100801A'])
df['OCF'] = df['C001000000'] / df['A001000000']
df['SOE'] = df['PropertyRightsNature']
df['Top1'] = df['ContrshrProportion']
df['Board'] = df['Boardsize']
df['Indep'] = df['IndDirectorRatio']
df['Dual'] = df['ConcurrentPosition']
df['Balance'] = df['SharesBalance']
df['EstablishDate'] = df['EstablishDate'].astype(str).str.strip()
df['est_year'] = pd.to_numeric(df['EstablishDate'].str[:4], errors='coerce')
df['Age'] = df['year'] - df['est_year']

CONT = ['Size', 'Lev', 'ROA', 'ROE', 'Growth', 'Cash', 'CashRatio',
        'TobinQ', 'lnMktCap', 'OCF', 'Top1', 'Balance', 'Age']
wins_info = {}
for c in CONT:
    s = df[c]
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    wins_info[c] = {'p1': float(lo), 'p99': float(hi), 'n': int(((s < lo) | (s > hi)).sum())}
    df[c] = s.clip(lo, hi)

rename = {
    'Stkcd': 'stkcd', 'ShortName': 'shortname', 'Accper': 'accper',
    'IndustryName': 'indname_csrc', 'IndustryCode': 'indcode_csrc',
    'IndustryNameC': 'indnameC', 'IndustryCodeC': 'indcodeC',
    'PROVINCE': 'province', 'CITY': 'city',
    'A001000000': 'total_assets', 'A002000000': 'total_liab',
    'A001101000': 'monetary_cap', 'A003000000': 'total_equity',
    'B001101000': 'revenue', 'B002000000': 'netprofit',
    'B002000101': 'netprofit_parent', 'C001000000': 'ocf',
    'F100801A': 'mktcap',
}
df = df.rename(columns=rename)

out_cols = ['stkcd', 'shortname', 'year', 'accper', 'indcode_csrc', 'indname_csrc',
            'indcodeC', 'indnameC', 'province', 'city', 'LISTINGDATE', 'EstablishDate',
            'total_assets', 'total_liab', 'monetary_cap', 'total_equity', 'revenue',
            'netprofit', 'netprofit_parent', 'ocf', 'mktcap',
            'Size', 'Lev', 'ROA', 'ROE', 'Growth', 'Cash', 'CashRatio',
            'TobinQ', 'lnMktCap', 'OCF', 'SOE', 'Top1', 'Board', 'Indep', 'Dual',
            'Balance', 'Age']
df = df.sort_values(['stkcd', 'year']).reset_index(drop=True)
df[out_cols].to_parquet(WS + r"\cleaned_panel_2019_2024.parquet", index=False)

print("\n公司数:", df['stkcd'].nunique(), " 观测:", len(df))
print(df['year'].value_counts().sort_index())
with open(WS + r"\clean_log_2024.json", 'w', encoding='utf-8') as f:
    json.dump({'清洗日志': [{'步骤': s, '剩余观测': int(n)} for s, n in log],
               '缩尾信息': wins_info,
               '最终公司数': int(df['stkcd'].nunique()),
               '最终观测数': int(len(df)),
               '年份分布': {int(k): int(v) for k, v in df['year'].value_counts().sort_index().items()}},
              f, ensure_ascii=False, indent=2)
print("完成。输出: cleaned_panel_2019_2024.parquet / clean_log_2024.json")
