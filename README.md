# 한국 소비자를 위한 LLM 핀테크 자문 감사

**한국 소비자를 위한 거대언어모델(LLM) 핀테크 자문 감사: 다중 모델·합성 페르소나 기반 실증 분석**
(Auditing Large Language Model FinTech Advice for Korean Consumers)

조천룡(Tianlong Zhao) · 성균관대학교 일반대학원 핀테크융합전공 · 『재무관리연구』 투고 예정

이 저장소에는 위 논문의 원고, 데이터, 분석 코드, 결과가 들어 있습니다.

## 연구 개요

- **페르소나:** Nemotron-Personas-Korea에서 연령대 × 성별 × 지역으로 층화 추출한 가상의 한국인 소비자 프로필 100개(실제 사람이 아님)
- **시나리오:** 한국 핀테크 소비자 상황 6개(S1 모바일 대출, S2 로보어드바이저, S3 암호화폐·레버리지, S4 오픈뱅킹, S5 스미싱, S6 BNPL 할부)
- **자문 모델:** Claude Opus 4.6, DeepSeek, Gemini 3 Flash, Kimi K2.5, GPT‑4o(OpenRouter 경유, 동일한 보호 지시문 적용)
- **응답:** 100 × 6 × 5 = 3,000개 셀 중 2,999개 응답(2026년 4월 30일 생성)
- **채점:**
  - 채점 모델(LLM-as-a-judge): 1차 GPT‑4o가 전체 응답을, 2차 Claude Sonnet 4.5가 층화 표본 510개 응답을 채점했습니다.
  - 한국어 어휘 지표: 기관·사기·위험·개인정보 관련 표현을 규칙 기반으로 검출합니다.
- **분석:**
  - 페르소나 군집 회귀와 페르소나 수준 순열검정
  - η²와 Benjamini–Hochberg FDR
  - 랜덤포레스트와 SHAP
  - 두 채점 모델 간 일치도와 측정 타당성 점검
- **사람 검증:** 수행하지 않았습니다. 후속 연구를 위한 제안 절차와 검증표만 포함되어 있습니다(원고 부록 C).

## 폴더 구성

원고 본문은 학술지 심사 중이라 이 저장소에 포함하지 않습니다. 사람 검증용 정답 키(`verification_key.csv`, `manual_validation_sample_key.csv`)는 검증자 눈가림을 위해 공개하지 않으며, `scripts/09_human_verification_sheet.py`와 `scripts/03_score_outputs.py`로 다시 만들 수 있습니다.

| 경로 | 내용 |
|---|---|
| `manuscript/figures_ko/`, `figures/` | 원고 그림(한글·영문) |
| `config/` | 시나리오 문구(영어 원문)와 기대 안전 요소, 자문 모델 목록, 루브릭 |
| `data/` | 페르소나 표본, 프롬프트, 모델 응답, 채점 결과, 2차 채점 표본, 사람 검증 제안용 검증표 |
| `results/` | 분석 결과 표와 그림(`revision/`, `revision2/`는 심사 대응 추가 분석, `ml_shap/`은 SHAP) |
| `scripts/` | 전체 파이프라인 스크립트(아래 표) |
| `docs/` | 사람 검증 판단 기준서(제안 절차용) |

## 스크립트

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| 1 | `00_sample_personas.py` | 페르소나 100개 추출: 연령대×성별×지역 층마다 최소 1명을 비례 배정한 뒤 100명을 넘으면 시드 42로 다시 무작위 추출 |
| 2 | `01_build_prompt_grid.py` | 페르소나 × 시나리오 프롬프트 생성 |
| 3 | `02_generate_outputs.py` | 자문 모델 응답 생성(API 필요) |
| 4 | `03_score_outputs.py` | 채점 모델로 루브릭 채점(API 필요) |
| 5 | `03b_response_features.py` | 한국어 어휘 지표 추출 |
| 6 | `04_analyze.py` | 회귀, 혼합효과, η², FDR |
| 7 | `05_ml_shap_analysis.py` | 랜덤포레스트·SHAP |
| 8 | `05_manual_validation_template.py` | 2차 채점용 층화 표본 추출 |
| 9 | `06_make_report.py` | 결과 요약 보고서(`results/REPORT.md`) |
| 10 | `07_manuscript_figures.py` | 원고 그림 생성 |
| 11 | `08_revision_checks.py`, `10_measurement_rq2_checks.py` | 심사 대응 추가 분석 |
| 12 | `09_human_verification_sheet.py` | 사람 검증 제안용 검증표 생성 |
| 13 | `11_docx_three_line_tables.py` | 원고 docx 표를 학술지 3선 표 서식으로 변환 |
| 14 | `12_presubmission_checks.py` | 투고 전 점검: 기관 언급 중복 집계 수정, 어휘 지표 미검출, Friedman 검정 집합, 부트스트랩 영분모, 유효 분모, 모델별 채점 일치도 |
| 15 | `13_presubmission_ml.py` | 투고 전 점검: 중복 없이 센 기관 언급 수로 표 10(교차검증 R²)과 그림 7(SHAP) 재추정 |

## 재현 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # OPENROUTER_API_KEY, OPENAI_API_KEY 입력(생성·채점 단계에만 필요)
```

이미 생성된 응답과 채점 결과가 `data/`에 있으므로, 분석만 다시 실행할 때는 API 키가 필요 없습니다.

```bash
python scripts/03b_response_features.py
python scripts/04_analyze.py --outdir results
python scripts/05_ml_shap_analysis.py --dataset results/analysis_dataset.csv --outdir results/ml_shap
python scripts/08_revision_checks.py
python scripts/10_measurement_rq2_checks.py
python scripts/07_manuscript_figures.py --lang ko --out manuscript/figures_ko
```

원고 docx 생성:

```bash
cd manuscript
pandoc manuscript_full_ko.md -o manuscript_full_ko.docx --columns=2000 --resource-path=. --reference-doc=kfma_reference.docx
python ../scripts/11_docx_three_line_tables.py manuscript_full_ko.docx
```

## 투고 전 점검 (2026-11-23)

보관된 응답과 채점 결과만으로 다시 계산하며 API 키가 필요 없습니다. 결과는 `results/presubmission/`에 저장됩니다.

```bash
python scripts/12_presubmission_checks.py
python scripts/13_presubmission_ml.py
```

- **기관 언급 수 중복 집계 수정:** `03b_response_features.py`는 이제 같은 글자에 겹쳐 맞는 패턴(예: `금융위원회`와 `금융위`)을 한 번으로 셉니다. 학술대회 발표본에 쓴 옛 방식은 `_count_matches_raw`로 남겨 두었습니다. 보관된 `data/response_features.csv`는 옛 방식의 값이므로, `03b`를 다시 실행하면 기관 언급 횟수가 바뀝니다(유효 응답 2,998개 중 730개).
- **어휘 지표:** 사전에 정의한 표현의 검출 여부이며 의미상의 언급률이 아닙니다. `lexical_human_review_sample.csv`는 사람 검토용 표본(106개 응답)입니다.
- **패키지 버전:** 재계산에 쓴 정확한 버전은 `requirements-lock.txt`에 있습니다.

## 유의 사항

- 페르소나는 합성 프로필이며, 실제 소비자의 행동이나 선택을 관찰한 것이 아닙니다.
- 점수는 LLM 채점 모델의 평가이며, 사람이 검증한 사실 오류율이 아닙니다.
- 모델 문자열은 날짜가 고정된 스냅숏이 아닌 OpenRouter 별칭이므로, 같은 설정으로 다시 생성해도 응답이 완전히 같지 않을 수 있습니다.
- API 키가 담긴 `.env`는 저장소에 포함하지 않습니다.
