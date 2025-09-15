<!-- quick result generated locally from remote aggregate metrics -->
# WMDP‑Bio (20% subset) — Llama 3.3 70B AWQ via vLLM

- Model: `llama-3.3-70b-awq` (vLLM OpenAI API, port 8002)
- Dataset: WMDP‑Bio, 20% sampled (n≈255)
- Seed: 123

## Accuracy (±95% CI)
- mcq_full: 0.714 [0.658, 0.769] (n=255)
- cloze_full: 0.714 [0.658, 0.769] (n=255)
- mcq_choices_only: 0.498 [0.436, 0.560] (n=255)

## MCQ vs Choices‑Only Difference
- Difference (mcq_full − choices_only): 0.216
- z ≈ 4.98, p ≈ 6.3e-7 (two‑sided)

Interpretation: Marked drop on choices‑only (Δ ≈ 0.216), highly significant, replicating the effect observed on WMDP‑Chem with a larger sample. Baseline MCQ and Cloze matched on this subset.

## Paths (remote)
- Logs: `~/cbrn-ai-hackathon/logs/wmdp_bio_llama33/`
- Aggregates: `~/cbrn-ai-hackathon/artifacts/results/wmdp_bio/`

Limitations: single model/seed, AWQ quantization, letter‑only responses, no tool use or CoT.

