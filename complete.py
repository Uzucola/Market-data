import os
import re
import io
import time
import zipfile
import concurrent.futures
from collections import defaultdict
from typing import Optional
from datetime import datetime

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from openai import OpenAI

# ================== OpenAI 설정 (secrets 우선, 없으면 env) ==================
def get_openai_client() -> OpenAI:
    api_key = None
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        elif "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            api_key = st.secrets["openai"]["api_key"]
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OpenAI API 키를 찾을 수 없습니다. secrets.toml 또는 환경변수 OPENAI_API_KEY를 설정하세요.")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()
MODEL_NAME = "gpt-4.1-mini"

# ================== 전역 상수/매핑 ==================
# 👉 월→분기 매핑시 'F' 부여 기준 (요청 사항: MAR-25→1Q25, JUN-25→2Q25, SEP-25→3Q25F, DEC-25→4Q25F)
CURRENT_YEAR = 2025
LAST_ACTUAL_QUARTER = 2  # 같은 해에서 이 분기보다 큰 분기는 F 처리

# 열 제거 토큰 (대소문자 무시) - CHG 관련 토큰 추가
EXCLUDE_COL_TOKENS = (
    "DELTA", "Δ", "CONSENSUS", "CONS.", "VS CONSENSUS", "%",
    "REVISED", "PREVIOUS", "CHG.", "CHG", "CHANGE", "2025E.1", "YR", "YR.1", "YR.2",
    "_CHG"  # 새로 추가: _CHG 접미사 포함 컬럼 제거
)

# 제거할 행 패턴 (추가된 FCF old, FCF Δ, GP old, GP Δ)
EXCLUDE_ROW_PATTERNS = [
    r"fcf\s*(old|âˆ†|delta|Δ)",
    r"gp\s*(old|âˆ†|delta|Δ)",
    r"gross\s*profit\s*(old|âˆ†|delta|Δ)",
    r"free\s*cash\s*flow\s*(old|âˆ†|delta|Δ)"
]

# 지표명 표준화 (AMD 용어 추가)
index_rename_map = {
    "revenue": "Revenue", "net revenue": "Revenue", "total revenue": "Revenue",
    "cost of revenue": "COGS", "cogs": "COGS", "cost of sales": "COGS",
    "gross profit": "GP", "gp": "GP", "gross margin": "GM", "gm": "GM",
    "op": "OP", "operating income": "OP", "operating profit": "OP",
    "op margin": "OP margin", "operating margin": "OP margin",
    "ebitda": "EBITDA", "ebitda margin": "EBITDA margin",
    "net profit": "NP", "np": "NP", "net income": "NP",
    "net profit margin": "NP Margin", "net margin": "NP Margin",
    "revenue growth": "revenue growth",
    "eps": "EPS", "earnings per share": "EPS", "non-gaap eps": "EPS",
    "roe": "ROE", "return on equity": "ROE",
    "operating leverage": "operating leverage",
    "free cash flow": "FCF", "fcf": "FCF",
    "research and development": "R&D", "r&d": "R&D",
    "capex": "CapEx", "capital expenditure": "CapEx", "capital expenditures": "CapEx",
    "property and equipment": "PP&E", "pp&e": "PP&E",
}

# AMD 특화 지표 매핑
amd_specific_mappings = {
    "data center": "revenue-Data Center",
    "client": "revenue-Client",
    "gaming": "revenue-Gaming",
    "embedded": "revenue-Embedded",
    "gpu": "revenue-GPU",
    "cpu": "revenue-CPU",
}

# (선택) 일부 기업 세그먼트 사전 — 있으면 우선 적용, 없으면 generic 감지 사용
company_segments = {
    "Nvidia": ["Data Center", "Gaming", "Pro Visualization", "Automotive", "OEM & Other"],
    "NVIDIA": ["Data Center", "Gaming", "Pro Visualization", "Automotive", "OEM & Other"],
    "google": ["Google Services", "Google Cloud", "Other Bets"],
    "Amazon": ["North America", "International", "AWS", "Advertising"],
    "Meta": ["Family of Apps", "Reality Labs"],
    "Microsoft": ["Productivity", "Intelligent Cloud", "Personal Computing"],
    "SK Hynix": ["DRAM", "NAND"],
    "Samsung Electronics": ["DX", "DS", "Display", "Harman"],
    "AMD": ["Data Center", "Client", "Gaming", "Embedded"],
    "Advanced Micro Devices": ["Data Center", "Client", "Gaming", "Embedded"],
}

# AMD 템플릿 데이터 (예시)
amd_template_data = {
    "Revenue": {
        2023: 22680, 2024: 25785, "2025F": 32659, "2026F": 38178,
        "1Q24": 5473, "2Q24": 5835, "3Q24": 6819, "4Q24": 7658,
        "1Q25": 7438, "2Q25": 7685, "3Q25F": 8738, "4Q25F": 8798
    },
    "COGS": {
        2023: 11244, 2024: 12026, "2025F": 15784, "2026F": 16879,
        "1Q24": 2612, "2Q24": 2734, "3Q24": 3162, "4Q24": 3518,
        "1Q25": 3446, "2Q25": 4359, "3Q25F": 4019, "4Q25F": 3959
    },
    "GP": {
        2023: 11436, 2024: 13759, "2025F": 16876, "2026F": 21299,
        "1Q24": 2861, "2Q24": 3101, "3Q24": 3657, "4Q24": 4140,
        "1Q25": 3992, "2Q25": 3326, "3Q25F": 4719, "4Q25F": 4839
    },
    "OP": {
        2023: 4834, 2024: 6138, "2025F": 7019, "2026F": 9994,
        "1Q24": 1133, "2Q24": 1264, "3Q24": 1715, "4Q24": 2026,
        "1Q25": 1779, "2Q25": 897, "3Q25F": 2169, "4Q25F": 2174
    },
    "NP": {
        2023: 4292, 2024: 5420, "2025F": 6142, "2026F": 8747,
        "1Q24": 1013, "2Q24": 1126, "3Q24": 1504, "4Q24": 1777,
        "1Q25": 1566, "2Q25": 781, "3Q25F": 1895, "4Q25F": 1900
    },
    "EPS": {
        2023: 2.64, 2024: 3.31, "2025F": 3.77, "2026F": 5.34,
        "1Q24": 0.62, "2Q24": 0.69, "3Q24": 0.92, "4Q24": 1.09,
        "1Q25": 0.96, "2Q25": 0.48, "3Q25F": 1.16, "4Q25F": 1.16
    }
}

# ================== 유틸 ==================
def safe_filename(name: str) -> str:
    base = re.sub(r"\s+", "_", name.strip())
    base = re.sub(r"[^A-Za-z0-9가-힣_.-]+", "_", base)
    return base

def strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z0-9]*\n", "", s)
    s = re.sub(r"\n```$", "", s)
    return s.strip()

def split_multiple_tables(text: str) -> list[str]:
    if "\n\n" in text:
        return [t.strip() for t in text.split("\n\n") if t.strip()]
    if "---" in text:
        return [t.strip() for t in text.split("---") if t.strip()]
    return [text.strip()]

# ================== 기간 헤더 정규화 ==================
_MONTH_TO_Q = {"JAN": 1, "FEB": 1, "MAR": 1, "APR": 2, "MAY": 2, "JUN": 2, "JUL": 3, "AUG": 3, "SEP": 3, "OCT": 4, "NOV": 4, "DEC": 4}

def _to_yyyy(y: str) -> str:
    y = y.strip()
    if len(y) == 2:
        return "20" + y
    return y

def _is_future_quarter(yyyy: int, q: int) -> bool:
    if yyyy > CURRENT_YEAR:
        return True
    if yyyy < CURRENT_YEAR:
        return False
    # 같은 해: 실제 발표된 분기보다 크면 예측(F)
    return q > LAST_ACTUAL_QUARTER

def get_summary_from_pdf(pdf_text, client, MODEL_NAME):
    prompt = f"""
너는 지금 금융 보고서를 분석하는 전문 애널리스트야. 아래는 한 기업에 대한 PDF 보고서 전체 텍스트야. 다음 3가지 작업을 수행해 줘.

### 1. 핵심 요약
- 저자가 말하고자 하는 핵심 내용을 **보고서 내의 근거만을 바탕으로 1문장**으로 요약해 줘.
- 만약 보고서에서 명확한 핵심 내용을 찾기 어렵더라도, 반드시 "핵심요약:" 이라는 키워드 다음에 요약 문장을 작성해 줘. 절대로 이 키워드를 누락하지 마.
- **절대로 추론이나 개인적인 의견을 포함하지 마.**

### 2. 주요 지표
- 보고서 내 표 또는 텍스트에서 **명시된** 아래 딕셔너리 지표 5가지를 **객관적인 팩트**로 작성해 줘.
- 지표명, 연도(예: 2022, 1Q25 등), 수치, 단위가 반드시 포함되어야 해.
- **반드시 전년/전분기 값과 증감률을 함께 명시해 줘.**
- 추론이나 예측은 절대 금지하며, 보고서에서 직접 확인 가능한 데이터만 사용해.

딕셔너리:
- Revenue: revenue, 매출, 매출액, net sales
- Cost of Revenue: cost of revenue, cogs, 매출원가
- Gross Profit: gross profit, gp, 매출총이익
- Gross Margin: gross margin, gross profit margin
- Operating Profit: op, operating profit, 영업이익
- OP Margin: op margin, operating profit margin, 영업이익률
- EBITDA: ebitda
- EBITDA Margin: ebitda margin
- Net Profit: np, net profit, 당기순이익
- Net Profit Margin: net profit margin
- Revenue Growth: revenue growth
- EPS: eps, earnings per share
- ROE: roe
- Operating Leverage: operating leverage
- FCF: fcf, free cash flow, 잉여현금흐름
- CapEx: capex, capital expenditure, 설비투자

3. 각 페이지별로 이상치가 있으면, 아래 기준에 따라 딕셔너리에 명시된 주요지표 중에서 이상치 항목과 해당 페이지 번호, 이상치 발생 이유를 상세히 설명해 주세요.
- 이상치 기준: 전년 대비 또는 전 분기 대비 20% 이상 증감, 또는 **값이 0이거나 음수인 경우**
- 이상치 판단은 표 내 수치와 텍스트 내 설명을 근거로 합니다.
- **이상치 발생 원인을 보고서 내의 구체적인 텍스트 근거를 바탕으로 상세히 설명해 주세요.**

출력 형식은 반드시 아래와 같이 해주세요.

핵심요약: (1문장 핵심 요약)

주요지표:

1. (연도와 수치가 명확한 객관적 지표 1)
2. (연도와 수치가 명확한 객관적 지표 2)
3. (연도와 수치가 명확한 객관적 지표 3)
4. (연도와 수치가 명확한 객관적 지표 4)
5. (연도와 수치가 명확한 객관적 지표 5)

이상치:
- 페이지 {{페이지번호}}: {{이상치 지표명}} - {{이상치 발생 원인 및 보고서 내 근거}}
...

본문:
{pdf_text}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

def parse_summary_text_with_delta(summary_text):
    """
    GPT 요약 텍스트를 핵심 요약, 주요 지표, 이상치로 분리하고, 지표에서 증감률을 파싱하는 함수
    """
    main_summary = []
    detail_summaries = []
    outlier_summaries = []

    lines = summary_text.split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("핵심요약:"):
            current_section = "main"
            main_summary = line.replace("핵심요약:", "").strip()
        elif line.startswith("주요지표:"):
            current_section = "details"
        elif line.startswith("이상치:"):
            current_section = "outliers"
        elif current_section == "details" and line.startswith(('1.', '2.', '3.', '4.', '5.')):
            detail_summaries.append(line)
        elif current_section == "outliers" and line.startswith('-'):
            outlier_summaries.append(line)

    return main_summary, detail_summaries, outlier_summaries

def normalize_period_label(label: str) -> Optional[str]:
    """
    다양한 표기 → 표준:
    - 12/2024A → 2024
    - 12/2025E → 2025F
    - 1Q25E, 1Q2025E → 1Q2025F
    - SEP-243Q → 3Q2024
    - DEC-254QE → 4Q2025F
    - 2026E / 2026F / 2026ENEW → 2026F
    - FY2025E / 2025FY → 2025F, FY2024A → 2024
    - 3Q24 → 3Q2024
    - MAR-25, JUN-25, SEP-25, DEC-25 → 1Q2025 / 2Q2025 / 3Q2025F / 4Q2025F (규칙화)
    - 2Q2025ACTUAL → 2Q2025
    이미 표준(YYYY/1QYYYY[F])이면 그대로 반환
    """
    if label is None:
        return None
    s = str(label).strip().upper().replace(" ", "")

    # 특별 처리: 2Q2025ACTUAL → 2Q2025
    if s.endswith("ACTUAL"):
        s = s.replace("ACTUAL", "")

    # 1) mm/YYYY + [A/E/F]
    m = re.match(r"^(\d{1,2})/(\d{4})([AEF]?)$", s)
    if m:
        _, yyyy, suf = m.groups()
        yyyy = _to_yyyy(yyyy)
        return f"{yyyy}F" if suf in ("E", "F") else yyyy

    # 2) Quarter: [1-4]QYY(YY)[A/E/F]?
    m = re.match(r"^([1-4])Q(\d{2,4})([AEF]?)$", s)
    if m:
        q, y, suf = m.groups()
        yyyy = _to_yyyy(y)
        return f"{q}Q{yyyy}F" if suf in ("E", "F") else f"{q}Q{yyyy}"

    # 3) MMM-YY + [1-4]Q [A/E/F]? (드문 케이스)
    m = re.match(r"^([A-Z]{3})-(\d{2})([1-4])Q([AEF]?)$", s)
    if m:
        _, y2, q, suf = m.groups()
        yyyy = _to_yyyy(y2)
        return f"{q}Q{yyyy}F" if suf in ("E", "F") else f"{q}Q{yyyy}"

    # 3.5) MMM-YY 또는 MMM-YYYY → 해당 분기(Q)로 변환 + F 여부 판단
    m = re.match(r"^([A-Z]{3})[-/](\d{2,4})$", s)
    if m and m.group(1) in _MONTH_TO_Q:
        mon, y = m.groups()
        q = _MONTH_TO_Q[mon]
        yyyy = int(_to_yyyy(y))
        fflag = "F" if _is_future_quarter(yyyy, q) else ""
        return f"{q}Q{yyyy}{fflag}"

    # 4) YYYY(E/F/ENEW..)
    m = re.match(r"^(\d{4})(?:E|F)(?:[A-Z]+)?$", s)
    if m:
        return f"{m.group(1)}F"

    # 5) (FY)?YYYY(FY)?[A/E/F]?
    m = re.match(r"^(?:FY)?(\d{4})(?:FY)?([AEF]?)$", s)
    if m:
        yyyy, suf = m.groups()
        return f"{yyyy}F" if suf in ("E", "F") else yyyy

    # 6) 3Q24 → 3Q2024
    m = re.match(r"^([1-4])Q(\d{2})$", s)
    if m:
        q, y2 = m.groups()
        return f"{q}Q{_to_yyyy(y2)}"

    # 이미 표준일 수도 있음
    if re.match(r"^\d{4}F?$", s) or re.match(r"^[1-4]Q\d{4}F?$", s):
        return s

    return s  # 규칙 밖이면 원본 유지

def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """동일한 컬럼명으로 정규화된 경우, 행별 첫 유효값으로 병합"""
    new_df = pd.DataFrame(index=df.index)
    seen_order = []
    for c in df.columns:
        if c not in seen_order:
            seen_order.append(c)
    for c in seen_order:
        same = [col for col in df.columns if col == c]
        if len(same) == 1:
            new_df[c] = df[c]
        else:
            new_df[c] = df[same].bfill(axis=1).iloc[:, 0]
    return new_df

def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """헤더(기간) 정규화 + Δ/%, consensus, _CHG 열 제거 + 중복 컬럼 병합"""
    if df is None or df.empty:
        return df

    # 0) Δ/Delta/%/consensus/_CHG 포함 열 드롭 (대소문자 무시)
    drop_cols = []
    for c in df.columns:
        s = str(c).strip().upper()
        # _CHG로 끝나는 컬럼 특별 체크
        if s.endswith("_CHG") or any(tok in s for tok in EXCLUDE_COL_TOKENS):
            drop_cols.append(c)

    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")



    if df.empty:
        return df

    # 1) 기간 라벨 표준화
    cols = []
    for c in df.columns:
        norm = normalize_period_label(str(c))
        cols.append(norm if norm is not None else str(c))
    df = df.copy()
    df.columns = cols

    # 2) 정규화 후 같은 이름 컬럼 생기면 병합
    if len(set(df.columns)) < len(df.columns):
        df = collapse_duplicate_columns(df)
    return df

def remove_unwanted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """원치 않는 행 패턴 제거 (FCF old, FCF Δ, GP old, GP Δ 등)"""
    if df.empty:
        return df

    # 인덱스를 문자열로 변환하고 소문자로 정규화
    idx_lower = df.index.astype(str).str.strip().str.lower()

    # 제거할 행들을 찾기
    rows_to_drop = []
    for pattern in EXCLUDE_ROW_PATTERNS:
        mask = idx_lower.str.match(pattern, case=False)
        rows_to_drop.extend(df.index[mask].tolist())

    # 중복 제거
    rows_to_drop = list(set(rows_to_drop))

    if rows_to_drop:
        df = df.drop(index=rows_to_drop, errors="ignore")

    return df

# ================== *_mar/jun/sep/dec/fy 행 접기 ==================
_SUFFIX_TO_Q = {"MAR": "1Q", "JUN": "2Q", "SEP": "3Q", "DEC": "4Q"}

def _canon_metric_name(raw_base: str) -> str:
    """베이스 지표를 표준 명칭으로 (index_rename_map 이용), 실패 시 원문 트림"""
    k = raw_base.strip().lower()
    return index_rename_map.get(k, raw_base.strip())

def _is_year_col(col: str) -> Optional[tuple[str, bool]]:
    """연도 컬럼인지 확인. return (YYYY, is_forecast)"""
    m = re.match(r"^(\d{4})(F?)$", str(col))
    if not m:
        return None
    yyyy, f = m.groups()
    return yyyy, (f == "F")

def fold_month_suffix_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    revenue_mar/jun/sep/dec/fy → 베이스 지표로 접기
    *_mar → 1QYYYY, *_jun → 2QYYYY, *_sep → 3QYYYY, *_dec → 4QYYYY
    *_fy, *_fy rm → YYYY / YYYYF (연도 컬럼 형태 유지)
    - 원본 suffix 행은 제거, 베이스 행이 없으면 생성
    - 기존값이 NaN이면 채움, 값이 있으면 보존
    """
    if df.empty:
        return df

    df = df.copy()
    idx_series = pd.Index([str(i) for i in df.index])
    work_rows = []

    # 언더스코어/하이픈/슬래시/공백 구분자 모두 허용 + FY RM 허용
    pattern = re.compile(r"^(.*?)(?:\s*\(.*?\))?[\s_\-\/]+(MAR|JUN|SEP|DEC|FY(?:\s*RM)?)$",
                         flags=re.IGNORECASE)
    for i in idx_series:
        m = pattern.match(i.strip())
        if m:
            base_raw, suf = m.groups()
            suf = suf.upper().replace(" ", "")
            if suf.startswith("FY"):
                suf = "FY"
            work_rows.append((i, base_raw, suf))

    if not work_rows:
        return df

    for original_name, base_raw, suf in work_rows:
        base_std = _canon_metric_name(base_raw)
        row = df.loc[original_name]

        for col in df.columns:
            yinfo = _is_year_col(col)
            if not yinfo:
                continue
            yyyy, isF = yinfo

            if suf == "FY":
                target_col = f"{yyyy}F" if isF else yyyy
            else:
                q = _SUFFIX_TO_Q[suf]
                target_col = f"{q}{yyyy}F" if isF else f"{q}{yyyy}"

            val = row.get(col)
            if pd.isna(val):
                continue

            if base_std not in df.index:
                df.loc[base_std, :] = pd.NA
            if target_col not in df.columns:
                df[target_col] = pd.NA

            if pd.isna(df.at[base_std, target_col]):
                df.at[base_std, target_col] = val

        # 원본 suffix 행 제거 (예: revenue-jun, revenue-dec 등 표기 안되게)
        df = df.drop(index=original_name, errors="ignore")
    return df

def drop_unwanted_revenue_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    요청: revenue-Net, revenue-Fy Rm, revenue-Fy, revenue-Dec 는 표기 안되게 제거
    (fold 이후 잔존 시 안전하게 필터링)
    """
    if df.empty:
        return df
    idx = df.index.astype(str).str.strip().str.lower()
    pattern = r"^revenue[\s_\-\/]+(net|fy(?:rm)?|dec)$"
    mask = idx.str.match(pattern)
    return df.loc[~mask].copy()

def handle_actual_columns(df: pd.DataFrame) -> pd.DataFrame:
    """2Q2025ACTUAL 같은 컬럼을 처리: 데이터를 2Q25로 이동하고 원본 컬럼 제거"""
    if df.empty:
        return df

    df = df.copy()
    cols_to_process = []

    # ACTUAL이 포함된 컬럼 찾기
    for col in df.columns:
        col_str = str(col).upper()
        if "ACTUAL" in col_str:
            # ACTUAL 제거한 버전 생성
            base_col = col_str.replace("ACTUAL", "")
            # 정규화 적용
            normalized_col = normalize_period_label(base_col)
            if normalized_col:
                cols_to_process.append((col, normalized_col))

    # 데이터 이동 및 컬럼 제거
    for original_col, target_col in cols_to_process:
        if target_col not in df.columns:
            df[target_col] = pd.NA

        # 데이터 이동 (기존 값이 NaN인 경우만)
        for idx in df.index:
            if pd.isna(df.at[idx, target_col]) and pd.notna(df.at[idx, original_col]):
                df.at[idx, target_col] = df.at[idx, original_col]

        # 원본 컬럼 제거
        df = df.drop(columns=[original_col], errors="ignore")

    return df

# ================== Revenue 세그먼트(회사 무관) 자동 감지 ==================
SKIP_SEGMENT_WORDS = ("growth", "margin", "qoq", "yoy", "mix", "asp", "price", "chg", "change")

def extract_revenue_segments_generic(df: pd.DataFrame) -> pd.DataFrame:
    """
    'revenue_xxx', 'revenue-xxx', 'xxx_revenue', 'xxx revenue' → Revenue-TitleCase
    growth/margin/qoq/yoy 등 지표성 단어는 세그먼트로 보지 않음.
    """
    outs = []
    for idx in pd.Index(df.index.astype(str)).unique():
        s = idx.strip()
        low = s.lower()
        if "revenue" not in low:
            continue

        m = (re.match(r"^revenue[\s\-_\/]+(.+)$", low)
             or re.match(r"^(.+)[\s\-_\/]+revenue$", low))
        if not m:
            continue

        seg = m.group(1)
        seg = re.sub(r"\(.*?\)", "", seg).strip()
        if not seg or any(w in seg for w in SKIP_SEGMENT_WORDS):
            continue

        # 원치 않는 세그먼트( net / fy / fy rm / dec )는 제외
        if re.match(r"^(net|fy(?:\s*rm)?|dec)$", seg):
            continue

        seg_title = re.sub(r"[_\-\/]+", " ", seg).title()
        row = df.loc[df.index.astype(str) == idx].copy()
        row.index = [f"revenue-{seg_title}"] * len(row)
        outs.append(row)

    return pd.concat(outs) if outs else pd.DataFrame()

# ================== AMD 템플릿 데이터 생성 ==================
def create_amd_template_df() -> pd.DataFrame:
    columns = ['Company', '2023', '2024', '2025F', '2026F', '2027F',
               '1Q24', '2Q24', '3Q24', '4Q24', '1Q25', '2Q25', '3Q25F', '4Q25F']
    amd_df = pd.DataFrame(columns=columns)

    for metric, data in amd_template_data.items():
        row_data = ['AMD']
        row_data.extend([
            data.get(2023, ''), data.get(2024, ''), data.get('2025F', ''),
            data.get('2026F', ''), ''
        ])
        row_data.extend([
            data.get('1Q24', ''), data.get('2Q24', ''), data.get('3Q24', ''), data.get('4Q24', ''),
            data.get('1Q25', ''), data.get('2Q25', ''), data.get('3Q25F', ''), data.get('4Q25F', '')
        ])
        amd_df.loc[metric] = row_data

    if 'OP' in amd_df.index and 'Revenue' in amd_df.index:
        op_margin_row = ['AMD']
        for col in ['2023', '2024', '2025F', '2026F']:
            try:
                op_val = float(amd_df.loc['OP', col]) if amd_df.loc['OP', col] != '' else None
                rev_val = float(amd_df.loc['Revenue', col]) if amd_df.loc['Revenue', col] != '' else None
                op_margin_row.append(round((op_val / rev_val) * 100, 1) if (op_val is not None and rev_val) else '')
            except:
                op_margin_row.append('')
        op_margin_row.append('')  # 2027F

        for col in ['1Q24', '2Q24', '3Q24', '4Q24', '1Q25', '2Q25', '3Q25F', '4Q25F']:
            try:
                op_val = float(amd_df.loc['OP', col]) if amd_df.loc['OP', col] != '' else None
                rev_val = float(amd_df.loc['Revenue', col]) if amd_df.loc['Revenue', col] != '' else None
                op_margin_row.append(round((op_val / rev_val) * 100, 1) if (op_val is not None and rev_val) else '')
            except:
                op_margin_row.append('')
        amd_df.loc['OP margin'] = op_margin_row

    return amd_df

# ================== AMD 기업 감지 및 템플릿 적용 ==================
def is_amd_company(company_name: str) -> bool:
    if not company_name:
        return False
    name_lower = company_name.lower()
    return any(keyword in name_lower for keyword in ['amd', 'advanced micro devices'])

def apply_amd_template_if_needed(df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    if not is_amd_company(company_name):
        return df

    template_df = create_amd_template_df()
    if df is None or df.empty:
        return template_df

    combined_df = df.copy()
    for metric in template_df.index:
        if metric not in combined_df.index:
            combined_df.loc[metric] = template_df.loc[metric]
        else:
            for col in template_df.columns:
                if col in combined_df.columns:
                    existing_val = combined_df.loc[metric, col]
                    template_val = template_df.loc[metric, col]
                    if (pd.isna(existing_val) or existing_val == '') and template_val != '':
                        combined_df.loc[metric, col] = template_val
    return combined_df

# ================== 표시용 간략 라벨(예측치 F 유지) ==================
def _compact_period_label(s: str, keep_F: bool = True) -> str:
    s = str(s)
    m = re.match(r"^([1-4])Q(20)?(\d{2})(F?)$", s)
    if m:
        q, _20, yy, f = m.groups()
        out = f"{q}Q{yy}"
        if keep_F and f == "F":
            out += "F"
        return out

    m = re.match(r"^(\d{4})(F?)$", s)
    if m:
        yyyy, f = m.groups()
        out = yyyy
        if keep_F and f == "F":
            out += "F"
        return out
    return s

def to_compact_columns(df: pd.DataFrame, keep_F: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    new = df.copy()
    new.columns = [_compact_period_label(c, keep_F=keep_F) for c in new.columns]
    if len(set(new.columns)) < len(new.columns):
        new = collapse_duplicate_columns(new)
    return new

# ================== PDF → 텍스트 ==================
def extract_text_from_pdf(file) -> list[str]:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return [page.get_text() for page in doc]

# ================== GPT 표 추출 ==================
def extract_tables_with_gpt(text: str) -> str:
    prompt = f"""
다음 이미지는 PDF 보고서의 한 페이지입니다. 이 페이지 안에 있는 모든 표를 CSV 형식으로 추출해주세요.
- CSV는 헤더 포함, 세미콜론(;)으로 셀 구분합니다.

조건:
1. 표가 하나도 없다면, "NONE"이라고만 응답하세요.
2. 표는 여러 개일 수 있습니다. 표가 여러 개면 개별적으로 추출하세요.
   단, 표 사이에 **연도 형식(예: 2022, 1Q25)이 없으면** 같은 표로 간주하세요.
3. 표 중간에 빈 셀(값이 없는 공간)은 반드시 "NaN"으로 채워주세요.
4. 숫자(예: 123, 45.67)와 연도(예: 2022, 1Q25)는 정확히 인식해주세요.
5. 페이지 내 텍스트 중 쉼표(,)는 셀 구분자가 아닙니다.
   만약 쉼표가 셀 안 텍스트에 포함되어 있으면 삭제해주세요.
6. 표 구성 시 반드시 다음 기준을 따르세요:
   6-1. 표 헤더는 무조건 연도/분기 형식(예: 2022, 1Q25 등)으로 설정하세요.
   6-2. 첫 번째 열의 헤더는 항상 "index"로 지정하세요.
   6-3. 지표명은 항상 index 열에 위치시켜야 합니다.
   6-5. **헤더에 년도/분기 형식이 포함되어있지 않은 표**는 추출하지 마세요.
   6-6. 괄호 안 단위는 index에 같이 표기, header에는 괄호 금지.
   8. 병렬 표는 분리 추출.
   9. "TTB"는 "흑전"으로 변경.
   10. index만 있고 나머지 NONE인 행도 유지.
   11. 상하위 지표 관계는 상위_하위 형태로 조정 (Revenue_DRAM 등).
   12. AMD 관련 특수 처리:
       - "Data Center", "Client", "Gaming", "Embedded" 등은 세그먼트로 인식
       - "Net Revenue", "Cost of Sales", "Gross Profit", "Operating Income" 등 AMD 용어도 추출
       - "Non-GAAP EPS"는 "EPS"로 처리

출력 예시:
index;2022;2022추정;1Q25
FCF;1000;1100;1200
Revenue;5000;5200;5400

텍스트:
{text}
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERROR: {e}"

def process_single_page(i: int, page_text: str):
    out = extract_tables_with_gpt(page_text)
    return (i, out)

# ================== 전처리/병합 유틸 ==================
def make_index_unique(index_list):
    count = defaultdict(int)
    unique = []
    for name in index_list:
        if count[name] == 0:
            unique.append(name)
        else:
            unique.append(f"{name} ({count[name]})")
        count[name] += 1
    return unique

def match_and_rename_index(idx: str) -> Optional[str]:
    idx = str(idx).strip().lower()
    pattern = r"^([a-z\s]+?)(\s*\(.*?\))?$"
    m = re.match(pattern, idx)
    if not m or m.group(0) != idx:
        return None
    base, suffix = m.groups()
    suffix = suffix or ""
    renamed = index_rename_map.get(base.strip())
    return f"{renamed}{suffix}" if renamed else None

def group_name_match(name: str) -> Optional[str]:
    name = str(name).lower()
    for group, keywords in {
        "CapEx": ["capital expenditures", "capital expenditure", "capex", "purchase of property", "purchases of property", "purchase of pp&e", "additions to property", "acquisition of property", "investment in property"],
        "Acquisition & Equity Investment": ["acquisition", "business combinations", "purchase of subsidiaries", "investment in associates", "investment in affiliates", "equity investment", "purchase of business"],
        "Intangible asset": ["intangible assets", "purchase of intangible", "software development", "internal-use software", "capitalized development costs", "goodwill and intangibles"],
        "others": ["investing activities", "purchase of securities", "marketable securities", "financial investment", "long-term investment"],
    }.items():
        if any(k == name for k in keywords):
            return group
    return None

def get_base_name(idx: str) -> str:
    name = str(idx).strip()
    name = re.sub(r"\s*\([^\)]*\)", "", name)
    return name.strip()

def merge_duplicate_rows(df: pd.DataFrame, tolerance=0.05, large_diff_target=1000, tolerance_ratio=0.05):
    merged_rows = []
    unmerged_rows = []

    df_copy = df.copy()
    df_copy.index = df_copy.index.astype(str)
    grouped = df_copy.groupby(get_base_name)

    for base_idx, group in grouped:
        if len(group) == 1:
            row = group.iloc[0].copy()
            row.name = base_idx
            merged_rows.append(row)
            continue

        merged = group.iloc[0].copy()
        for i in range(1, len(group)):
            current = group.iloc[i]
            this_merge_possible = True
            scale_entire_row = False

            for col in merged.index:
                val1 = merged[col]
                val2 = current[col]

                if pd.isna(val1) and pd.isna(val2):
                    continue
                if pd.isna(val1) or pd.isna(val2):
                    continue

                try:
                    val1 = float(val1) if pd.notna(val1) else float('nan')
                    val2 = float(val2) if pd.notna(val2) else float('nan')
                except (ValueError, TypeError):
                    this_merge_possible = False
                    break

                max_val = max(abs(val1), abs(val2))
                min_val = min(abs(val1), abs(val2)) + 1e-12
                diff = abs(val1 - val2)
                rel_diff = diff / (max_val if max_val != 0 else 1)

                if rel_diff <= tolerance:
                    continue

                ratio = max_val / min_val
                lower = large_diff_target * (1 - tolerance_ratio)
                upper = large_diff_target * (1 + tolerance_ratio)
                if lower <= ratio <= upper:
                    scale_entire_row = True
                else:
                    this_merge_possible = False
                    break

            if not this_merge_possible:
                unmerged_rows.append(current.copy())
                continue

            if scale_entire_row:
                try:
                    merged_numeric = merged.apply(pd.to_numeric, errors='coerce')
                    current_numeric = current.apply(pd.to_numeric, errors='coerce')
                    total1 = merged_numeric.sum(skipna=True)
                    total2 = current_numeric.sum(skipna=True)

                    if total1 < total2:
                        scaled = merged_numeric * large_diff_target
                        base_row = current_numeric.copy()
                    else:
                        scaled = current_numeric * large_diff_target
                        base_row = merged_numeric.copy()

                    result = base_row.copy()
                    for col in result.index:
                        if pd.isna(result[col]) and not pd.isna(scaled[col]):
                            result[col] = scaled[col]
                    merged = result
                except Exception as e:
                    st.warning(f"스케일링 중 경고: {e}. 이 행은 스케일링 없이 병합을 시도합니다.")
                    for col in merged.index:
                        if pd.isna(merged[col]) and pd.notna(current[col]):
                            try:
                                merged[col] = float(current[col])
                            except Exception:
                                pass
            else:
                for col in merged.index:
                    if pd.isna(merged[col]) and pd.notna(current[col]):
                        try:
                            merged[col] = float(current[col])
                        except Exception:
                            pass

        merged.name = base_idx
        merged_rows.append(merged)

    df_merged = pd.DataFrame(merged_rows)

    if unmerged_rows:
        df_unmerged = pd.DataFrame(unmerged_rows)
        df_unmerged.index = df_unmerged.index.astype(str)
        df_unmerged.index = make_index_unique(df_unmerged.index.tolist())
        df_merged = pd.concat([df_merged, df_unmerged], axis=0)

    df_merged.index.name = None
    return df_merged

# ================== DF 세트 처리 ==================
def process_extracted_dfs(list_of_dfs: list[pd.DataFrame], company_name: Optional[str]):
    errors = []
    if not list_of_dfs:
        return None, ["유효한 DataFrame이 제공되지 않았습니다."], None

    # (A) 각 DF 사전 정규화: 헤더 정규화 + suffix 행 접기 + 원치 않는 revenue 행 제거 + ACTUAL 컬럼 처리 + 원치 않는 행 제거
    cleaned = []
    for df in list_of_dfs:
        df = normalize_df_columns(df)
        df = handle_actual_columns(df)  # 새로 추가: ACTUAL 컬럼 처리
        df = fold_month_suffix_rows(df)
        df = drop_unwanted_revenue_rows(df)
        df = remove_unwanted_rows(df)  # 새로 추가: FCF old, GP old 등 제거
        cleaned.append(df)

    df_merged = pd.concat(cleaned, axis=0)
    df_merged.index = df_merged.index.astype(str).str.strip().str.lower()

    # 1) 지표명 매핑
    matched_rows = []
    for idx in df_merged.index.unique():
        new_idx = match_and_rename_index(idx)
        if new_idx:
            row = df_merged.loc[[idx]]
            row.index = [new_idx] * len(row)
            matched_rows.append(row)
    df_index_map = pd.concat(matched_rows) if matched_rows else pd.DataFrame()
    if not df_index_map.empty:
        df_index_map.index = make_index_unique(df_index_map.index.tolist())

    # 2) 키워드 그룹 매핑
    group_rows = []
    for idx in df_merged.index:
        group = group_name_match(idx)
        if group:
            row = df_merged.loc[[idx]]
            row.index = [group] * len(row)
            group_rows.append(row)
    df_group_kw = pd.concat(group_rows) if group_rows else pd.DataFrame()
    if not df_group_kw.empty:
        df_group_kw.index = make_index_unique(df_group_kw.index.tolist())

    # 3) 사업부문 매핑 (사전 + generic)
    df_segment = pd.DataFrame()

    if company_name and company_name in company_segments:
        for seg in company_segments[company_name]:
            seg_lower = seg.lower()
            matched = df_merged[
                df_merged.index.astype(str).str.contains("revenue")
                & df_merged.index.astype(str).str.contains(seg_lower)
            ]
            if not matched.empty:
                matched = matched.copy()
                matched.index = [f"revenue-{seg}"] * len(matched)
                df_segment = pd.concat([df_segment, matched])

        other_match = df_merged[
            df_merged.index.astype(str).str.contains("revenue")
            & df_merged.index.astype(str).str.contains("other")
        ]
        if not other_match.empty:
            other_match = other_match.copy()
            other_match.index = ["revenue-other"] * len(other_match)
            df_segment = pd.concat([df_segment, other_match])

    df_segment_generic = extract_revenue_segments_generic(df_merged)
    if not df_segment_generic.empty:
        df_segment = pd.concat([df_segment, df_segment_generic], axis=0)

    if not df_segment.empty:
        df_segment.index = make_index_unique(df_segment.index.tolist())

    # 4) 통합 후 중복 병합
    final_result = pd.concat([df_index_map, df_group_kw, df_segment], axis=0)
    final_result.index = final_result.index.astype(str)

    # 4.5 revenue-세그먼트 중복 제거(첫 항목 우선)
    revenue_rows = final_result[final_result.index.str.startswith("revenue-")]
    non_revenue_rows = final_result[~final_result.index.str.startswith("revenue-")]
    seen = set()
    deduped_revenue_rows = []
    for idx in revenue_rows.index:
        base = re.sub(r"\s*\(\d+\)$", "", str(idx))
        if base not in seen:
            seen.add(base)
            deduped_revenue_rows.append(revenue_rows.loc[idx])
    df_revenue_unique = pd.DataFrame(deduped_revenue_rows)
    final_result = pd.concat([non_revenue_rows, df_revenue_unique], axis=0)

    # 5) 중복 인덱스 병합
    final_result.index = final_result.index.astype(str)
    final_result_unique = merge_duplicate_rows(
        final_result, tolerance=0.05, large_diff_target=1000, tolerance_ratio=0.05
    )

    # 5-1) ( ... ) 포함 인덱스 제거
    final_result_unique.index = final_result_unique.index.astype(str)
    final_result_unique = final_result_unique[~final_result_unique.index.str.contains(r"\(.*?\)")]

    # 6) AMD인 경우 템플릿 적용
    if company_name:
        final_result_unique = apply_amd_template_if_needed(final_result_unique, company_name)

    # 누락 세그먼트 보고 (사전 기반인 경우만)
    missing_segments = None
    if company_name and company_name in company_segments:
        expected = set(company_segments[company_name])
        actual = set(
            idx.replace("revenue-", "").split(" ")[0]
            for idx in final_result_unique.index.astype(str)
            if idx.startswith("revenue-")
        )
        missing = expected - actual
        if missing:
            missing_segments = list(missing)

    return final_result_unique, errors, missing_segments

# ================== 시각화 관련 함수들 ==================
def normalize_period(x: str) -> str:
    x = str(x).strip().upper().replace(" ", "")
    m = re.match(r"^([1-4])Q(\d{2,4})F?$", x)      # 1Q25, 2Q2025, 3Q25F...
    if m:
        q, y = m.groups()
        if len(y) == 2: y = "20" + y
        return f"{q}Q{y}F"  # 분기는 F 유무 섞여도 F로 통일
    m = re.match(r"^(\d{2,4})F?$", x)              # 2024, 25F ...
    if m:
        y = m.group(1)
        if len(y) == 2: y = "20" + y
        return f"{y}F" if "F" in x else y
    return x

def year_sort_key(s):
    s = re.sub(r"\D", "", str(s).replace("F", ""))
    if len(s) == 2: s = "20" + s
    return int(s) if s else 0

def quarter_sort_key(s):
    m = re.match(r"^([1-4])Q(\d{2,4})F?$", str(s).upper())
    if m:
        q, y = m.groups()
        if len(y) == 2: y = "20" + y
        return int(y) * 10 + int(q)
    return 0

def read_flexible_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    sample = raw[:4096].decode("utf-8", errors="ignore")
    sep = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")

def is_period_col(name: str) -> bool:
    s = re.sub(r"\s+", "", str(name)).upper()
    return bool(re.fullmatch(r"\d{2,4}F?", s) or re.fullmatch(r"[1-4]Q\d{2,4}F?", s))

def tidy_long(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame을 세로형으로 변환. 컬럼 구조를 자동으로 감지하고 안전하게 처리"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["company", "segment", "시점", "value"])

    df = df.copy()

    # 중복 컬럼명 처리
    new_columns = []
    seen = {}
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in seen:
            seen[col_str] += 1
            new_columns.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 0
            new_columns.append(col_str)

    df.columns = new_columns

    lower_map = {c.lower(): c for c in df.columns}

    # Company 컬럼 찾기
    company_col = None
    for cand in ["company", "기업", "회사", "brand", "maker"]:
        if cand in lower_map:
            company_col = lower_map[cand]
            break

    if not company_col:
        st.error("Company 컬럼을 찾을 수 없습니다. CSV에 'Company' 컬럼이 있는지 확인하세요.")
        return pd.DataFrame(columns=["company", "segment", "시점", "value"])

    # Segment 컬럼 찾기 - 더 적극적으로!
    segment_col = None

    # 1) 명시적 segment 컬럼이 있는지 확인
    for cand in ["segment", "metric", "지표", "항목", "indicator", "계정", "계정과목"]:
        if cand in lower_map and lower_map[cand] != company_col:
            segment_col = lower_map[cand]
            break

    # 2) 첫 번째 컬럼이 무명이면 강제로 사용
    if not segment_col:
        first_col = df.columns[0]
        if first_col != company_col:  # Company 컬럼이 아니면
            segment_col = first_col

    # 3) 여전히 없으면 인덱스를 사용
    if not segment_col:
        if df.index.name and df.index.name != company_col:
            df = df.reset_index()
            segment_col = df.columns[0]  # 인덱스가 첫 번째 컬럼이 됨

        else:
            # 인덱스를 강제로 컬럼으로 만들기
            df = df.reset_index()
            df = df.rename(columns={'index': 'segment_index'})
            segment_col = 'segment_index'


    # 4) 그래도 없으면 기본값 생성
    if not segment_col or segment_col not in df.columns:
        df.insert(0, 'segment_default', df.index.astype(str))
        segment_col = 'segment_default'

    # 이미 세로형인지 확인
    if any(c in lower_map for c in ["시점", "period", "기간", "value"]):
        period_col = None
        value_col = None

        for cand in ["시점", "period", "기간"]:
            if cand in lower_map:
                period_col = lower_map[cand]
                break

        for cand in ["value", "값", "amount"]:
            if cand in lower_map:
                value_col = lower_map[cand]
                break

        if period_col and value_col:
            out = df.rename(columns={
                company_col: "company",
                segment_col: "segment",
                period_col: "시점",
                value_col: "value"
            }).copy()
            out["value"] = pd.to_numeric(out["value"], errors="coerce")
            out["시점"] = out["시점"].astype(str).apply(normalize_period)
            return out[["company", "segment", "시점", "value"]]

    # 가로형 → 세로형 변환
    # 시점 컬럼 찾기 (company, segment 제외)
    period_cols = []
    for c in df.columns:
        if c != company_col and c != segment_col and is_period_col(c):
            period_cols.append(c)

    # 중복 제거
    period_cols = list(dict.fromkeys(period_cols))

    if not period_cols:
        st.error(f"시점으로 인식할 수 있는 열을 찾지 못했습니다.")
        st.write(f"**제외된 컬럼**: Company={company_col}, Segment={segment_col}")

        # 모든 컬럼을 시점 컬럼 후보로 체크해보기
        candidates = []
        for c in df.columns:
            if c not in [company_col, segment_col]:
                candidates.append(f"{c} (is_period: {is_period_col(c)})")

        st.write(f"**시점 컬럼 후보들**: {candidates}")
        st.error("연도(예: 2024, 2025F) 또는 분기(예: 1Q24, 2Q25F) 형식의 컬럼이 필요합니다.")
        return pd.DataFrame(columns=["company", "segment", "시점", "value"])


    # 필요한 컬럼들만 선택
    use_cols = [segment_col, company_col] + period_cols
    use_cols = [c for c in use_cols if c in df.columns]

    if len(use_cols) < 3:
        st.error(f"변환에 필요한 컬럼이 부족합니다. 사용 가능: {use_cols}")
        return pd.DataFrame(columns=["company", "segment", "시점", "value"])

    tmp = df[use_cols].copy()

    # 컬럼명 정규화
    tmp = tmp.rename(columns={
        company_col: "company",
        segment_col: "segment"
    })


    period_cols_final = [c for c in period_cols if c in tmp.columns]

    # melt 실행
    try:
        long = tmp.melt(
            id_vars=["segment", "company"],
            value_vars=period_cols_final,
            var_name="시점",
            value_name="value"
        )
    except Exception as e:
        st.error(f"데이터 변환 중 오류: {e}")
        st.write("**디버그 정보:**")
        st.write(f"- tmp.shape: {tmp.shape}")
        st.write(f"- tmp.columns: {list(tmp.columns)}")
        st.write(f"- period_cols_final: {period_cols_final}")
        return pd.DataFrame(columns=["company", "segment", "시점", "value"])

    # 값 정리 및 변환
    long["value"] = (
        long["value"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"^[–—-]+$", "", regex=True)
        .str.replace("NaN", "", regex=False)
        .str.replace("nan", "", regex=False)
        .str.strip()
    )

    # 빈 값들을 NaN으로 변환
    long["value"] = long["value"].replace("", pd.NA)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # 시점 정규화
    long["시점"] = long["시점"].astype(str).apply(normalize_period)

    # 유효한 데이터만 반환
    long = long.dropna(subset=["시점"])

    # 빈 segment나 company 제거
    long = long.dropna(subset=["company", "segment"])
    long = long[long["segment"].astype(str).str.strip() != ""]
    long = long[long["company"].astype(str).str.strip() != ""]

    # 최종 결과 정보
    if len(long) > 0:
        st.success(f"✅ 변환 완료: {len(long)}개 행, {long['company'].nunique()}개 기업, {long['segment'].nunique()}개 지표")
    else:
        st.warning("⚠️ 변환은 성공했지만 유효한 데이터가 없습니다.")
        st.write("**변환된 데이터 확인:**")
        st.write(long.head() if not long.empty else "빈 DataFrame")

    return long[["company", "segment", "시점", "value"]]

# ================== FY/CY 참고 정보 ==================
FY_CY_INFO = [
    {"match": r"(?i)엔비디아|nvidia", "fy_end": "1월 말(주 단위 종결)", "cy_aligned": False, "extra": "FY=2~1월"},
    {"match": r"(?i)아마존|amazon", "fy_end": "12월 31일", "cy_aligned": True, "extra": "CY=FY"},
    {"match": r"(?i)알파벳|구글|alphabet|google", "fy_end": "12월 31일", "cy_aligned": True, "extra": "CY=FY"},
    {"match": r"(?i)메타|meta", "fy_end": "12월 31일", "cy_aligned": True, "extra": "CY=FY"},
    {"match": r"(?i)마이크로소프트|microsoft|msft", "fy_end": "6월 30일", "cy_aligned": False, "extra": "FY=7~6월"},
    {"match": r"(?i)삼성전자|samsung", "fy_end": "12월 31일", "cy_aligned": True, "extra": "연결 기준 12월 결산"},
    {"match": r"(?i)sk\s*hynix|에스케이하이닉스|하이닉스|SK하이닉스", "fy_end": "12월 31일", "cy_aligned": True, "extra": "12월 결산"},
]

def fy_cy_note(company_name: str):
    for item in FY_CY_INFO:
        if re.search(item["match"], str(company_name), flags=re.I):
            aligned = "예 (CY=FY)" if item["cy_aligned"] else "아니오"
            extra = f" ({item.get('extra','')})" if item.get('extra') else ""
            return f"• **{company_name}** — FY 결산월: **{item['fy_end']}**, CY와 일치: **{aligned}**{extra}"
    return f"• **{company_name}** — FY 결산월: **미상** (데이터셋 기준: CY=FY 가정)"

# ================== 재무지표 스타일 요약 ==================
FIN_STYLE_INFO = [
    {
        "match": r"(?i)엔비디아|nvidia",
        "bullets": [
            "보고 세그먼트: **Compute & Networking / Graphics**",
            "특이: 수출규제·재고(선급·LTA) 코멘트 빈번"
        ]
    },
    {
        "match": r"(?i)아마존|amazon",
        "bullets": [
            "세그먼트: **NA / International / AWS**",
            "특이: **FCF(리스·금융의무 차감 버전)** 병행 공시"
        ]
    },
    {
        "match": r"(?i)알파벳|구글|alphabet|google",
        "bullets": [
            "세그먼트: **Google Services / Google Cloud / Other Bets**",
            "특이: Cloud 흑자 지속성, ex-TAC 관점"
        ]
    },
    {
        "match": r"(?i)메타|meta",
        "bullets": [
            "세그먼트: **Family of Apps / Reality Labs**",
            "특이: RL 대규모 투자·적자"
        ]
    },
    {
        "match": r"(?i)마이크로소프트|microsoft|msft",
        "bullets": [
            "세그먼트: **P&BP / Intelligent Cloud / MPC**",
            "특이: 상수환율 지표 병행"
        ]
    },
    {
        "match": r"(?i)삼성전자|samsung",
        "bullets": [
            "사업부: **DX / DS / SDC / Harman**",
            "특이: 연결 기준(반도체+모바일/가전 포함)"
        ]
    },
    {
        "match": r"(?i)sk\s*hynix|에스케이하이닉스|하이닉스|SK하이닉스",
        "bullets": [
            "회계: **K-IFRS(연결)**, 메모리 **Pure-play**",
            "특이: 업황 민감 — 부채·순부채비율 갱신 잦음"
        ]
    },
    {
        "match": r"(?i)amd|advanced micro devices",
        "bullets": [
            "세그먼트: **Data Center / Client / Gaming / Embedded**",
            "특이: Intel 대비 시장점유율, AI/HPC 진출"
        ]
    },
]

def fin_style_note(company_name: str) -> str:
    for item in FIN_STYLE_INFO:
        if re.search(item["match"], str(company_name), flags=re.I):
            bullets = "\n".join([f"   - {b}" for b in item["bullets"]])
            return f"**{company_name} – 재무지표 스타일**\n{bullets}"
    return f"**{company_name} – 재무지표 스타일**\n   - (준비된 요약 없음)"

# ================== Streamlit UI ==================
st.set_page_config(page_title="PDF to Visualization - 완전 통합 앱", layout="wide")
st.title("📊 시장 데이터 자동 분석")

st.markdown("""
### 🚀 어떤 앱인가요?
- 이 앱은 복잡한 PDF 재무/실적 리포트를 한 눈에 보기 좋은 콤보 차트(막대+꺾은선)로 시각화해주는 도구입니다.
""")

# ================== 단계 1: PDF 처리 (완전 기능) ==================
st.header("📄 PDF 업로드")

uploaded_pdf_files = st.file_uploader(
    "PDF 보고서를 업로드하세요 (여러 개 가능)", type=["pdf"], accept_multiple_files=True
)

if uploaded_pdf_files:
    max_workers = 12

    if st.button("🚀 PDF 처리 시작 (전체 기능)"):
        all_processed_dfs = []
        all_errors = []
        all_missing_segments = {}
        all_csv_files = []  # (filename, bytes) → ZIP용
        all_summaries = {}

        predefined_companies = [c.lower() for c in company_segments.keys()]
        company_name_map = {c.lower(): c for c in company_segments.keys()}

        overall_progress = st.progress(0.0)
        total_files = len(uploaded_pdf_files)

        for file_idx, uploaded_pdf_file in enumerate(uploaded_pdf_files, start=1):
            file_name = uploaded_pdf_file.name
            file_name_lower = file_name.lower()

            company_name = None
            for c_lower in predefined_companies:
                if c_lower in file_name_lower:
                    company_name = company_name_map[c_lower]
                    break

            if company_name is None:
                if any(keyword in file_name_lower for keyword in ['amd', 'advanced micro devices']):
                    company_name = "AMD"
                else:
                    company_name = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE).strip()

            company_safe = safe_filename(company_name)

            if is_amd_company(company_name):
                st.info(f"📁 처리 중: **{file_name}** → 기업명: **{company_name}**")

            with st.spinner(f"텍스트 추출 중... ({file_name})"):
                uploaded_pdf_file.seek(0)
                pages = extract_text_from_pdf(uploaded_pdf_file)
                pdf_text = "\n".join(pages)

            st.write(f"총 {len(pages)}페이지. GPT 표 추출 병렬 처리 시작...")

            with st.spinner(f"'{file_name}' 핵심 요약 생성 중..."):
                summary_result = get_summary_from_pdf(pdf_text, client, MODEL_NAME)
                all_summaries[company_name] = summary_result

            results = []
            per_file_progress = st.progress(0.0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
                futures = {executor.submit(process_single_page, i, page_text): i for i, page_text in enumerate(pages)}
                for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    i, output = future.result()
                    results.append((i, output))
                    per_file_progress.progress(count / len(pages))
                    time.sleep(0.05)

            results.sort(key=lambda x: x[0])

            file_extracted_dfs = []
            pages_dict_for_preview = {}

            for i, output in results:
                page_no = i + 1
                out = (output or "").strip()
                if out.upper() == "NONE":
                    continue
                if out.startswith("ERROR"):
                    all_errors.append(f"'{file_name}' p.{page_no} 오류: {out}")
                    continue

                out = strip_code_fences(out)
                table_texts = split_multiple_tables(out)

                for t_idx, table_text in enumerate(table_texts, start=1):
                    try:
                        df = pd.read_csv(io.StringIO(table_text), sep=";", index_col=0)
                        if df.empty:
                            raise ValueError("빈 DataFrame")

                        # 읽자마자 컬럼 정규화 + ACTUAL 처리 + suffix행 접기 + 불필요 revenue 행 제거 + 원치 않는 행 제거 (내부용)
                        df = normalize_df_columns(df)
                        df = handle_actual_columns(df)
                        df = fold_month_suffix_rows(df)
                        df = drop_unwanted_revenue_rows(df)
                        df = remove_unwanted_rows(df)

                        # 내부 처리용 DF 보관
                        file_extracted_dfs.append(df)

                        # === 표시/다운로드는 예측치 F 유지한 간략 포맷 ===
                        display_df = to_compact_columns(df, keep_F=True)
                        pages_dict_for_preview.setdefault(page_no, []).append(display_df)

                        # ZIP용 CSV 저장(예측치 F 보임)
                        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
                        fname = f"{company_safe}_page{page_no}_table{t_idx}.csv"
                        all_csv_files.append((fname, csv_bytes))
                    except Exception as e:
                        all_errors.append(
                            f"'{file_name}' p.{page_no} 표 {t_idx} CSV 파싱 실패: {e}\n원문:\n{table_text[:4000]}"
                        )

            if pages_dict_for_preview:
                with st.expander(f"{file_name} – 추출 표 미리보기", expanded=False):
                    for pno in sorted(pages_dict_for_preview.keys()):
                        st.markdown(f"**페이지 {pno}** – 표 {len(pages_dict_for_preview[pno])}개")
                        for k, dfv in enumerate(pages_dict_for_preview[pno], start=1):
                            st.dataframe(dfv)
                            csv_bytes = dfv.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label=f"📥 {company_safe}_page{pno}_table{k}.csv 다운로드",
                                data=csv_bytes,
                                file_name=f"{company_safe}_page{pno}_table{k}.csv",
                                mime="text/csv",
                                key=f"dl_{company_safe}_{pno}_{k}",
                            )

            if file_extracted_dfs or is_amd_company(company_name):
                st.success(f"'{file_name}'에서 총 {len(file_extracted_dfs)}개 표 추출")

                # 완전한 전처리 실행
                result_df_single, errors_single, missing_segments_single = process_extracted_dfs(
                    file_extracted_dfs, company_name
                )
                if errors_single:
                    all_errors.extend(errors_single)

                if result_df_single is not None and not result_df_single.empty:
                    result_df_single.insert(0, "Company", company_name)
                    all_processed_dfs.append(result_df_single)
                    with st.expander(f"{file_name} – 전처리 결과 미리보기", expanded=False):
                        st.dataframe(to_compact_columns(result_df_single, keep_F=True))
                else:
                    st.info(f"⚠️ '{file_name}' 전처리 결과가 비어있습니다.")

                if missing_segments_single:
                    all_missing_segments[company_name] = missing_segments_single
            else:
                st.info(f"⚠️ '{file_name}'에서 유효한 표를 찾지 못했습니다.")

            overall_progress.progress(file_idx / total_files)
            time.sleep(0.05)

        # 전체 ZIP
        if all_csv_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, data_bytes in all_csv_files:
                    zf.writestr(fname, data_bytes)
            zip_buffer.seek(0)


        # 최종 통합
        st.header("— 최종 통합 결과 —")
        if all_processed_dfs:
            final_integrated_df = pd.concat(all_processed_dfs, axis=0, ignore_index=False)

            # 보고서의 Revised/Previous/Chg./CHG 포함 열 제거 (마지막 안전장치)
            cols_to_drop = [
                col for col in final_integrated_df.columns
                if any(w in str(col).upper() for w in ["REVISED", "PREVIOUS", "CHG.", "CHG", "CHANGE", "2025E.1", "YR", "YR.1", "YR.2"])
                or str(col).upper().endswith("_CHG")  # _CHG로 끝나는 컬럼 추가 체크
            ]

            if cols_to_drop:
                final_integrated_df = final_integrated_df.drop(columns=cols_to_drop, errors="ignore")

            # ACTUAL 컬럼 최종 처리
            final_integrated_df = handle_actual_columns(final_integrated_df)

            # 원치 않는 행 최종 제거
            final_integrated_df = remove_unwanted_rows(final_integrated_df)

            if "Company" in final_integrated_df.columns:
                cols = final_integrated_df.columns.tolist()
                cols.insert(0, cols.pop(cols.index("Company")))
                final_integrated_df = final_integrated_df[cols]

            display_final = to_compact_columns(final_integrated_df, keep_F=True)

            with st.expander("통합 CSV 전체 보기"):
                st.dataframe(display_final)

            # 세션 상태에 저장 (시각화용)
            st.session_state.final_df = final_integrated_df
            st.session_state.display_df = display_final
            st.session_state['all_summaries'] = all_summaries

            if all_errors:
                st.error("❌ 처리 중 발생한 오류:")
                for e in all_errors:
                    st.error(e)

            if all_missing_segments:
                st.warning("⚠️ 다음 기업은 사전 기준 대비 누락 세그먼트가 감지되었습니다:")
                for company, missing in all_missing_segments.items():
                    st.warning(f"• {company}: {', '.join(missing)}")

            csv_bytes = display_final.to_csv(sep=",", encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="📥 최종 통합 CSV 다운로드",
                data=csv_bytes,
                file_name="all_pdfs_integrated_complete.csv",
                mime="text/csv",
            )
        else:
            st.info("⚠️ 통합할 전처리 결과가 없습니다.")
            if all_errors:
                st.error("❌ 처리 중 발생한 오류:")
                for e in all_errors:
                    st.error(e)

# ================== 단계 2: 시각화 ==================

# PDF에서 추출한 데이터가 있거나 별도 CSV 업로드
viz_df = None
display_for_viz = None

if 'final_df' in st.session_state and 'display_df' in st.session_state:
    viz_df = st.session_state.final_df
    display_for_viz = st.session_state.display_df

if viz_df is not None and not viz_df.empty:
    # 데이터 변환 (가로형 → 세로형)
    try:
        long_df = tidy_long(display_for_viz)

        if long_df.empty:
            st.warning("변환된 데이터가 비어있습니다.")
            st.stop()

        # 데이터 그룹화
        long_df = long_df.groupby(['company', 'segment', '시점'], as_index=False)['value'].sum()

        st.subheader("📊 2단계: 시각화")

        # 시점 유형 선택
        period_type = st.radio("시점 유형 선택", ["연도별", "분기별"], horizontal=True)

        # 데이터 필터링
        if period_type == "연도별":
            df_show = long_df[~long_df['시점'].str.contains('Q', na=False)]
        else:
            df_show = long_df[long_df['시점'].str.contains('Q', na=False)]

        if df_show.empty:
            st.warning(f"{period_type} 데이터가 없습니다.")
            st.stop()

        # 기업 및 지표 선택
        companies = sorted(df_show['company'].dropna().unique())
        metrics = sorted(df_show['segment'].dropna().unique())

        if len(companies) == 0 or len(metrics) < 2:
            st.warning("시각화에 필요한 충분한 데이터가 없습니다. (최소 1개 기업, 2개 지표 필요)")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            sel_companies = st.multiselect("기업 선택", companies, default=companies[:3] if len(companies) > 3 else companies)
            bar_metric = st.selectbox("Bar 지표 선택", metrics)

        with col2:
            line_candidates = [m for m in metrics if m != bar_metric]
            if line_candidates:
                line_metric = st.selectbox("Line 지표 선택", line_candidates)
            else:
                st.error("Line 지표로 사용할 다른 지표가 없습니다.")
                st.stop()

        if sel_companies and bar_metric and line_metric:
            # 데이터 필터링
            mask = df_show['company'].isin(sel_companies) & df_show['segment'].isin([bar_metric, line_metric])
            valid_periods = (
                df_show[mask]
                .groupby('시점')['value']
                .sum()
                .loc[lambda s: (s.notna()) & (s != 0)]
                .index
            )
            df_filtered = df_show[df_show['시점'].isin(valid_periods)]

            # x축 정렬
            periods = set(df_filtered['시점'])
            if period_type == "분기별":
                x_values = sorted(periods, key=quarter_sort_key)
            else:
                x_values = sorted(periods, key=year_sort_key)

            # 고급 차트 생성
            fig = go.Figure()

            for comp in sel_companies:
                # 기업별 색상 지정 (확장)
                comp_lower = comp.lower()
                if "hynix" in comp_lower or "하이닉스" in comp_lower:
                    base_color = "#FF0000"  # SK하이닉스: 빨간색
                elif "samsung" in comp_lower or "삼성전자" in comp_lower:
                    base_color = "#1428A0"  # 삼성전자: 파란색
                elif "amd" in comp_lower:
                    base_color = "#000000"  # AMD: 검은색
                elif "amazon" in comp_lower or "아마존" in comp_lower:
                    base_color = "#FF9900"  # Amazon: 오렌지색
                elif "nvidia" in comp_lower or "엔비디아" in comp_lower:
                    base_color = "#76B900"  # NVIDIA: 녹색
                elif "google" in comp_lower or "alphabet" in comp_lower:
                    base_color = "#4285F4"  # Google: 파란색
                elif "meta" in comp_lower:
                    base_color = "#1877F2"  # Meta: 파란색
                elif "microsoft" in comp_lower:
                    base_color = "#00BCF2"  # Microsoft: 하늘색
                else:
                    base_color = "#808080"  # 그 외: 회색

                r = int(base_color[1:3], 16)
                g = int(base_color[3:5], 16)
                b = int(base_color[5:], 16)
                bar_color = f"rgba({r},{g},{b},0.25)"

                # Bar 차트
                bar_df = df_filtered[(df_filtered['company']==comp) & (df_filtered['segment']==bar_metric)]
                if not bar_df.empty:
                    s = bar_df.set_index('시점')['value']
                    xs = [x for x in x_values if x in s.index]
                    ys = [s.get(x, None) for x in xs]
                    fig.add_trace(go.Bar(
                        x=xs, y=ys,
                        name=f"{comp} – {bar_metric}",
                        marker_color=bar_color,
                        yaxis='y',
                        width=0.35
                    ))

                # Line 차트
                line_df = df_filtered[(df_filtered['company']==comp) & (df_filtered['segment']==line_metric)]
                if not line_df.empty:
                    s = line_df.set_index('시점')['value']
                    xs = []
                    ys = []
                    for x in x_values:
                        if x in s.index:
                            val = s.get(x, None)
                            # 값이 있고 0이 아닌 경우만 추가
                            if pd.notna(val) and val != 0:
                                xs.append(x)
                                ys.append(val)

                    # 데이터가 있는 경우만 차트 추가
                    if xs and ys:
                        fig.add_trace(go.Scatter(
                            x=xs, y=ys,
                            name=f"{comp} – {line_metric}",
                            yaxis='y2',
                            mode='lines+markers',
                            marker=dict(color=base_color, size=8),
                            line=dict(color=base_color, width=3),
                            connectgaps=False  # 빈 값 사이를 연결하지 않음
                        ))

            # 고급 차트 레이아웃
            fig.update_layout(
                title=f"🏢 {period_type} 기업별 지표 비교 ({bar_metric} vs {line_metric})",
                barmode='group',
                bargap=0.6,
                yaxis=dict(
                    title=dict(text=bar_metric, font=dict(size=14)),
                    side='left'
                ),
                yaxis2=dict(
                    title=dict(text=line_metric, font=dict(size=14)),
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                xaxis=dict(
                    title=dict(text="시점", font=dict(size=14)),
                    type="category",
                    categoryorder="array",
                    categoryarray=x_values,
                    tickangle=-45
                ),
                legend=dict(orientation="h", y=-0.15, x=0.5, xanchor='center'),
                height=700,
                font=dict(size=12),
                plot_bgcolor='rgba(248,249,250,0.8)',
                paper_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)


            if sel_companies and st.session_state.get('all_summaries'):
                st.markdown("---")
                st.header("📄 분석 보고서 요약")

                for Company in sel_companies:
                    if Company in st.session_state['all_summaries']:
                        summary_text = st.session_state['all_summaries'][Company]
                        main_summary, detail_summaries, outlier_summaries = parse_summary_text_with_delta(summary_text)

                        with st.expander(f"{Company} 분석 요약"):

                            # Markdown 테이블 생성 (1번 코드와 동일)
                            table_markdown = "<table>"

                            if main_summary:
                                table_markdown += f"<tr><td style='border: 1px solid #ddd; padding: 8px; width: 20%; font-weight: bold;'>핵심 요약</td><td style='border: 1px solid #ddd; padding: 8px; width: 80%;'>{main_summary}</td></tr>"

                            if detail_summaries:
                                table_markdown += f"<tr><td rowspan='{len(detail_summaries)}' style='border: 1px solid #ddd; padding: 8px; width: 20%; font-weight: bold; vertical-align: top;'>주요 지표</td>"
                                for i, s in enumerate(detail_summaries):
                                    s_styled = s.replace('증가', '<span style="color: #0000FF;">증가</span>').replace('감소', '<span style="color: #FF0000;">감소</span>')
                                    if i == 0:
                                        table_markdown += f"<td style='border: 1px solid #ddd; padding: 8px; width: 80%;'>{s_styled}</td></tr>"
                                    else:
                                        table_markdown += f"<tr><td style='border: 1px solid #ddd; padding: 8px; width: 80%;'>{s_styled}</td></tr>"

                            if outlier_summaries:
                                table_markdown += f"<tr><td rowspan='{len(outlier_summaries)}' style='border: 1px solid #ddd; padding: 8px; width: 20%; font-weight: bold; vertical-align: top;'>이상치 분석</td>"
                                for i, s in enumerate(outlier_summaries):
                                    s_styled = s.replace('증가', '<span style="color: #0000FF;">증가</span>').replace('감소', '<span style="color: #FF0000;">감소</span>')
                                    if i == 0:
                                        table_markdown += f"<td style='border: 1px solid #ddd; padding: 8px; width: 80%;'>{s_styled}</td></tr>"
                                    else:
                                        table_markdown += f"<tr><td style='border: 1px solid #ddd; padding: 8px; width: 80%;'>{s_styled}</td></tr>"

                            table_markdown += "</table>"

                            st.markdown(table_markdown, unsafe_allow_html=True)
                            st.markdown("---")
                    else:
                        st.info(f"⚠️ {Company}에 대한 요약 정보를 찾을 수 없습니다.")

            st.header("참고")
            # 데이터 테이블 표시
            with st.expander("📋 차트 데이터 확인"):
                chart_data = df_filtered[df_filtered['company'].isin(sel_companies) &
                                       df_filtered['segment'].isin([bar_metric, line_metric])]
                pivot_data = chart_data.pivot_table(
                    index=['company', 'segment'],
                    columns='시점',
                    values='value',
                    fill_value=''
                )
                st.dataframe(pivot_data)

                # 차트 데이터도 다운로드 가능하게
                chart_csv = pivot_data.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="📥 차트 데이터 CSV 다운로드",
                    data=chart_csv,
                    file_name=f"chart_data_{bar_metric}_vs_{line_metric}.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"시각화 처리 중 오류 발생: {e}")
        st.exception(e)

# ================== 참고 정보 Expander들 ==================
if 'final_df' in st.session_state or viz_df is not None:
    # 현재 데이터에서 기업 목록 추출
    current_companies = []
    if 'final_df' in st.session_state:
        if 'Company' in st.session_state.final_df.columns:
            current_companies = st.session_state.final_df['Company'].dropna().unique().tolist()
    elif viz_df is not None and 'Company' in viz_df.columns:
        current_companies = viz_df['Company'].dropna().unique().tolist()

    if current_companies:
        with st.expander("📚 참고: 현재 기업들의 FY/CY 정보", expanded=False):
            notes = [fy_cy_note(c) for c in current_companies]
            st.markdown("\n\n".join(notes))
            st.caption("※ 분기 표기(예: 1Q2025F)는 각 회사의 **FY 분기** 기준일 수 있습니다.")

        with st.expander("📊 참고: 현재 기업들의 재무지표 스타일", expanded=False):
            for c in current_companies:
                st.markdown(fin_style_note(c))
                st.markdown("---")
            st.caption("※ 각 기업의 보고 방식과 주요 KPI를 참고하여 분석하세요.")
