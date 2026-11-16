"""
Build a blinded human-verification workbook for institutional and legal claims.

Two strata, reported separately (do not extrapolate stratum B to an overall error rate):
  A  random:       3 responses per model x scenario cell from the 2,998 judged responses
  B  disagreement: responses in the 510-response second-judge subsample where the two LLM
                   judges disagree on the hallucination flag or differ by >= 2 points on
                   Korean contextual accuracy (excluding responses already in A)

Candidate claims (institutions, phone numbers, laws, named services/apps) are extracted with
regular expressions so the coder does not have to find them; the coder records whether each
claim exists, whether its role/contact details are correct, and whether it fits the situation.
Extraction over-captures on purpose; irrelevant rows are marked "해당 없음".

Outputs (data/human_verification/):
  verification_sheet.xlsx   blinded workbook (no model names, no judge scores)
  verification_key.csv      item_id -> stratum, model, judge scores (keep away from coders)

Usage:
    python scripts/09_human_verification_sheet.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "human_verification"
SEED = 2026
PER_CELL = 3
KEYS = ["persona_id", "scenario_id", "model_name"]

SCENARIO_KO = {"S1": "모바일 대출", "S2": "로보어드바이저", "S3": "암호화폐·레버리지",
               "S4": "오픈뱅킹", "S5": "스미싱", "S6": "BNPL 할부"}

ORG = re.compile(
    r"(?<![가-힣])([가-힣]{1,15}(?:위원회|진흥원|감독원|소비자원|공사|공단|센터|협회|재단|은행|증권"
    r"|협의회|조정위원회|청|처|원))(?![가-힣])"
    r"|\b(KISA|FSS|FSC|KFTC|KDIC|KCB|NICE)\b"
)
ORG_STOP = {"전문가원", "직원", "회원", "지원", "병원", "연구원", "상담원", "금액원", "만원", "천원", "억원",
            "원", "처", "청", "센터", "은행", "증권", "공사", "공단", "재단", "협회", "주거래은행",
            "해당은행", "거래은행", "각은행", "시중은행", "인터넷은행", "고객센터", "콜센터", "상담센터",
            "연락처", "판매원", "상담처", "거래처", "사용처", "문의처", "신고처", "가입처"}
PHONE = re.compile(r"(?<![\d,.])(?:☎\s*)?(1\d{3}-\d{4}|1\d{3}|1[1-3]\d|0\d{1,2}-\d{3,4}-\d{4})"
                   r"(?![\d,.%]|\s*(?:만|원|년|개|세|일|명|회|건|배|시간|분))")
LAW = re.compile(r"(?<![가-힣])([가-힣·]{2,25}(?:법|법률|시행령))(?![가-힣])")
LAW_STOP = {"방법", "불법", "합법", "편법", "사용법", "해결법", "대처법", "확인법", "관리법", "계산법",
            "투자법", "설정법", "신고법", "활용법", "이용법"}
SERVICE = re.compile(r"[‘'\"“]([^’'\"”\n]{2,30})[’'\"”]\s*(?:\([^)]{0,30}\)\s*)?(?:앱|어플|서비스|제도|플랫폼|시스템|사이트|홈페이지)")


def extract_claims(text: str) -> list[tuple[str, str, str]]:
    found: dict[tuple[str, str], str] = {}

    def add(kind: str, value: str, start: int, end: int) -> None:
        value = value.strip()
        if not value or (kind, value) in found:
            return
        ctx = text[max(0, start - 60): min(len(text), end + 60)].replace("\n", " ")
        found[(kind, value)] = ctx

    for m in ORG.finditer(text):
        v = m.group(1) or m.group(2)
        if v in ORG_STOP or len(v) < 3 and not v.isupper():
            continue
        add("기관", v, m.start(), m.end())
    for m in PHONE.finditer(text):
        add("전화번호", m.group(1), m.start(), m.end())
    for m in LAW.finditer(text):
        if m.group(1) not in LAW_STOP:
            add("법률", m.group(1), m.start(), m.end())
    for m in SERVICE.finditer(text):
        add("서비스·앱", m.group(1), m.start(), m.end())
    return [(k, v, c) for (k, v), c in found.items()]


def persona_summary(r: pd.Series) -> str:
    parts = [f"{r.get('std_age', '')}세" if pd.notna(r.get("std_age")) else "",
             str(r.get("std_sex", "")), str(r.get("std_region", "")),
             str(r.get("std_occupation", "")), str(r.get("std_household", ""))]
    return ", ".join(p for p in parts if p and p != "nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "results/analysis_dataset.csv")
    df = df.dropna(subset=["korean_context_accuracy"])
    out = pd.read_csv(ROOT / "data/model_outputs.csv")[KEYS + ["user_prompt"]]
    scen = pd.read_csv(ROOT / "config/scenarios.csv")[["scenario_id", "scenario_text"]]

    a = pd.concat([g.sample(n=min(PER_CELL, len(g)), random_state=SEED)
                   for _, g in df.groupby(["model_name", "scenario_id"])])
    a = a.assign(stratum="A_random")

    j2 = pd.read_csv(ROOT / "data/validation_judge2.csv")
    m = df.merge(j2[KEYS + ["korean_context_accuracy", "hallucinated_institution_or_law"]],
                 on=KEYS, suffixes=("", "_judge2"))
    hdis = m["hallucinated_institution_or_law_judge2"].notna() & (
        m["hallucinated_institution_or_law_judge2"] != m["hallucinated_institution_or_law"])
    kdis = (m["korean_context_accuracy_judge2"] - m["korean_context_accuracy"]).abs() >= 2
    b = m[hdis | kdis].copy()
    b["disagreement"] = (hdis[hdis | kdis].map({True: "환각", False: ""}) + " " +
                         kdis[hdis | kdis].map({True: "한국맥락", False: ""})).str.strip()
    in_a = set(map(tuple, a[KEYS].values))
    b = b[[tuple(x) not in in_a for x in b[KEYS].values]].assign(stratum="B_disagreement")

    items = pd.concat([a, b], ignore_index=True)
    items = items.sample(frac=1, random_state=SEED).reset_index(drop=True)  # blind the order
    items["item_id"] = [f"H{i + 1:03d}" for i in range(len(items))]
    items = items.merge(scen, on="scenario_id", how="left")

    # key file (not for coders)
    key_cols = ["item_id", "stratum", "disagreement"] + KEYS + [
        "korean_context_accuracy", "hallucinated_institution_or_law", "evidence_hallucination",
        "korean_context_accuracy_judge2", "hallucinated_institution_or_law_judge2"]
    items.reindex(columns=key_cols).to_csv(OUT / "verification_key.csv", index=False, encoding="utf-8-sig")

    wb = Workbook()
    guide = wb.active
    guide.title = "안내"
    lines = [
        "사람 검증 시트 (블라인드: 모델명과 판사 점수는 숨김)",
        "",
        "1. '응답' 시트에서 응답 전문을 읽고, '주장' 시트의 각 행(기관·전화번호·법률·서비스)을 공식 자료와 대조합니다.",
        "2. 공식 자료: 해당 기관 공식 홈페이지, 법제처 국가법령정보센터(law.go.kr), 정부 대표 누리집 등. 출처 URL을 기록합니다.",
        "3. 자동 추출이므로 기관·법률이 아닌 행은 '실재 여부'에 '해당 없음'을 고릅니다. 빠진 주장은 행을 추가해 적습니다.",
        "4. 판단 기준은 docs/human_verification_guide.md(판단 기준서)를 따릅니다.",
        "5. A층(무작위)과 B층(판사 불일치)은 따로 집계합니다. 응답 시트의 '층' 열은 검증 순서 안내용이며, 판단에 영향을 주지 않도록 A층부터 진행합니다.",
        "",
        f"응답 수: {len(items)} (A층 {int((items.stratum == 'A_random').sum())}, B층 {int((items.stratum == 'B_disagreement').sum())})",
    ]
    for i, l in enumerate(lines, 1):
        guide.cell(row=i, column=1, value=l)
    guide.column_dimensions["A"].width = 120

    wrap = Alignment(wrap_text=True, vertical="top")
    head_fill = PatternFill("solid", fgColor="DCE5DF")

    ws = wb.create_sheet("응답")
    rcols = ["item_id", "층", "시나리오", "질문", "페르소나 요약", "응답 전문",
             "사실 오류(없음/있음/판단 불가)", "구체성(1-3)", "정확성(1-3)", "상황 적합성(1-3)", "메모"]
    ws.append(rcols)
    for _, r in items.iterrows():
        ws.append([r.item_id, "A" if r.stratum == "A_random" else "B",
                   f"{r.scenario_id} {SCENARIO_KO[r.scenario_id]}", r.scenario_text,
                   persona_summary(r), r.response_text, "", "", "", "", ""])
    widths = [8, 5, 14, 40, 30, 90, 16, 10, 10, 12, 30]
    for c, w in zip("ABCDEFGHIJK", widths):
        ws.column_dimensions[c].width = w
    dv_err = DataValidation(type="list", formula1='"없음,있음,판단 불가"', allow_blank=True)
    dv_123 = DataValidation(type="list", formula1='"1,2,3"', allow_blank=True)
    ws.add_data_validation(dv_err)
    ws.add_data_validation(dv_123)
    n = len(items) + 1
    dv_err.add(f"G2:G{n}")
    dv_123.add(f"H2:J{n}")

    cs = wb.create_sheet("주장")
    ccols = ["item_id", "유형", "표현", "문맥", "실재 여부", "업무 범위·연락처 정확성",
             "이 상황에 적절한 안내인가", "공식 출처 URL", "메모"]
    cs.append(ccols)
    n_claims = 0
    for _, r in items.iterrows():
        for kind, value, ctx in extract_claims(str(r.response_text)):
            cs.append([r.item_id, kind, value, ctx, "", "", "", "", ""])
            n_claims += 1
    for c, w in zip("ABCDEFGHI", [8, 10, 22, 70, 14, 18, 18, 35, 30]):
        cs.column_dimensions[c].width = w
    dv_exist = DataValidation(type="list", formula1='"실재,부존재,명칭 오류,판단 불가,해당 없음"', allow_blank=True)
    dv_acc = DataValidation(type="list", formula1='"정확,부정확,해당 없음"', allow_blank=True)
    dv_fit = DataValidation(type="list", formula1='"적절,부분 적절,부적절,해당 없음"', allow_blank=True)
    for dv in (dv_exist, dv_acc, dv_fit):
        cs.add_data_validation(dv)
    m_ = n_claims + 1
    dv_exist.add(f"E2:E{m_}")
    dv_acc.add(f"F2:F{m_}")
    dv_fit.add(f"G2:G{m_}")

    for sheet in (ws, cs):
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.fill = head_fill
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = wrap
        sheet.freeze_panes = "B2"

    wb.save(OUT / "verification_sheet.xlsx")
    print(f"items={len(items)} (A={int((items.stratum == 'A_random').sum())}, "
          f"B={int((items.stratum == 'B_disagreement').sum())}), claims={n_claims}")


if __name__ == "__main__":
    main()
