# Results Template

Fill in after running `make aggregate`.

Overall:
- Models: (list)
- Tasks: (list)
- Items: (N)

Tables:

Exploitable (choices‑only consensus):
- exploitable % (majority of models correct in choices‑only): ____ %

Position bias:
- accuracy delta across seeds (A/B/C/D positions):
  - Seed1 vs Seed2: A: __, B: __, C: __, D: __

Heuristics:
- Longest‑answer accuracy: __ %
- Position‑bias rate (first/last): __ %
- Alphabetical preference rate: __ %

MCQ ↔ Cloze gap (95% CI):
- mean delta: __
- CI: [__, __]

Paraphrase consistency (95% CI):
- n pairs: __
- consistency: __
- CI: [__, __]
- figure: `artifacts/figs/paraphrase_consistency.png`

Perturbation fragility (flip rate, 95% CI):
- n pairs: __
- flip rate: __
- CI: [__, __]
- figure: `artifacts/figs/perturbation_fragility.png`

Paraphrase delta accuracy (orig − variants, 95% CI):
- n pairs: __
- mean delta: __
- CI: [__, __]

Paired Tests (McNemar):
- b (orig correct, variants wrong): __
- c (orig wrong, variants correct): __
- p-value (exact binomial): __
- FDR-adjusted q-value: __ (BH, α=0.05)

Multiple References (if available):
- exact match rate (95% CI): __ [__, __]
- normalized match rate (95% CI): __ [__, __]

Power Analysis (two-proportion, normal approx):
- target effect (Δ): __
- required N per group (α=0.05, 1-β=0.8): n1=__, n2=__

Abstention & Overconfidence:
- abstention rate: __ %
- overconfidence rate: __ %

Runtime / Cost:
- GPU hours: __ h
- Estimated GPU cost: $ __
- API spend (if any): $ __
