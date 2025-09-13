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
- Alphabetical preference rate: __ %

MCQ ↔ Cloze gap (95% CI):
- mean delta: __
- CI: [__, __]

Abstention & Overconfidence:
- abstention rate: __ %
- overconfidence rate: __ %

Runtime / Cost:
- GPU hours: __ h
- Estimated GPU cost: $ __
- API spend (if any): $ __

