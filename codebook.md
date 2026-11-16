# Codebook

## Unit of analysis

One row equals one LLM response to one synthetic persona in one FinTech scenario.

## Key identifiers

- `persona_id`: synthetic persona identifier
- `scenario_id`: standardized FinTech scenario
- `model_name`: advice-generating LLM
- `judge_model`: LLM used to score the output

## Main outcomes

- `riskiness_score`: 1–5, higher = riskier/worse
- `suitability_score`: 1–5, higher = better tailored to persona/scenario
- `disclosure_completeness`: 1–5, higher = more complete disclosure
- `consumer_protection_warning`: 0/1
- `korean_context_accuracy`: 1–5, higher = better Korea-specific fit
- `hallucinated_institution_or_law`: 0/1, higher = worse
- `actionability_score`: 1–5, higher = more actionable
- `respectful_tone`: 1–5, higher = more respectful

## Persona covariates

Scripts attempt to standardize columns as:

- `std_age`
- `age_group`
- `std_sex`
- `std_region`
- `std_district`
- `std_education`
- `std_occupation`
- `std_industry`
- `std_income`
- `std_household`
- `std_housing`
- `std_persona_text`

Because dataset schemas may change, inspect the sampled CSV after running `00_sample_personas.py`.
