# Analysis next steps after connecting Hugging Face

I confirmed the Hugging Face account connection and accessed the dataset metadata for `nvidia/Nemotron-Personas-Korea`.

Remote Hugging Face Jobs execution could not run because the connected account returned:

> 402 Payment Required — Pre-paid credit balance is insufficient.

The included notebook therefore runs the same analysis locally or in Google Colab. It streams the Hugging Face dataset directly, so it does not require downloading the full dataset first.

## Recommended pilot

1. Run the notebook with `N_PERSONAS = 100` and `N_GENERATION_LIMIT = 50`.
2. Check whether the column-standardization step correctly maps age, sex, region, education, occupation, income, and persona text.
3. Inspect 20 generated responses manually.
4. Run LLM-as-judge scoring.
5. Export `analysis_dataset.csv`, `summary_by_scenario.csv`, and regression outputs.

## Recommended main study

- 2,000 personas
- 6 scenarios
- 2 models
- 24,000 model responses
- 500 manually validated scored responses

## Main regression model

Outcome ~ age_group + sex + region + education + income_proxy + household_type + scenario fixed effects + model fixed effects

## Strongest expected paper result

The most publishable result is likely not a single bias claim. It will likely be a scenario-by-scenario audit showing where LLM financial advice is safer and where it fails:
- phishing/smishing: likely high warning quality
- crypto/leverage: likely mixed risk disclosure
- mobile loans/BNPL: likely weaker debt-risk framing
- open banking: likely generic privacy advice and possible Korean contextual gaps
