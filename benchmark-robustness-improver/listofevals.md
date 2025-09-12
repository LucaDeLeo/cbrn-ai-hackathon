# Bio/AI **LLM** benchmarks & evals — sorted by **relevance to CBRN**

## Direct CBRN / dual-use safety evaluations
1) **WMDP (Weapons of Mass Destruction Proxy)** — MCQ dataset targeting hazardous chem/bio/rad/nuke knowledge.
   → https://www.wmdp.ai/

2) **UK/US AI (Security) Safety Institute – Advanced AI Evaluations** — official pre-deployment tests for chemical & biological capabilities and safeguard effectiveness (incl. OpenAI o1).
   → https://www.aisi.gov.uk/work/advanced-ai-evaluations-may-update
   → https://www.aisi.gov.uk/work/pre-deployment-evaluation-of-openais-o1-model

3) **OpenAI Preparedness Framework (Biology/CBRN thresholds)** — capability thresholds & evaluation program for severe bio/chem risks.
   → https://openai.com/index/updating-our-preparedness-framework/
   → PDF: https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf
   → Context: “Preparing for future AI capabilities in biology”  https://openai.com/index/preparing-for-future-ai-capabilities-in-biology/

4) **METR (formerly ARC Evals)** — evaluations of dangerous model capabilities incl. bio-risk assistance & agent tests.
   → https://metr.org/blog/2023-12-04-metr-announcement/
   → https://metr.org/blog/2023-08-01-new-report/

5) **RAND: “Toward Comprehensive Benchmarking of the Biological & Chemical Knowledge of LLMs”** — cross-model testing (incl. WMDP) using AISI’s Inspect framework.
   → Report page: https://www.rand.org/pubs/working_papers/WRA3797-1.html
   → PDF: https://www.rand.org/content/dam/rand/pubs/working_papers/WRA3700/WRA3797-1/RAND_WRA3797-1.pdf

6) **MLCommons AILuminate** — independent safety benchmark with physical-hazard categories (e.g., weaponization prompts).
   → Overview: https://mlcommons.org/ailuminate/
   → Benchmark page: https://ailuminate.mlcommons.org/benchmarks/
   → GitHub: https://github.com/mlcommons/ailuminate

7) **“Forbidden Science: Dual-Use AI Challenge & Scientific Refusal Tests”** — open dual-use benchmark focusing on refusal vs over-restriction.
   → https://arxiv.org/pdf/2502.06867

8) **Anthropic – Biosecurity evaluations & ASL-3 deployment measures** — public write-ups on bio evals and protections for Claude.
   → https://www.anthropic.com/news/strategic-warning-for-ai-risk-progress-and-insights-from-our-frontier-red-team
   → https://red.anthropic.com/2025/biorisk/

---

## Wet-lab protocol & troubleshooting (procedural capability; **adjacent to CBRN**)
9) **BioLP-bench** — find/fix failures in wet-lab protocols and assess procedural reasoning.
   → https://www.biorxiv.org/content/10.1101/2024.08.21.608694v3.full-text

10) **LAB-Bench** — 2.4k+ biology research questions across figure/table reading, database navigation, **ProtocolQA**, cloning scenarios, sequence ops.
    → Paper: https://arxiv.org/abs/2407.10362
    → GitHub: https://github.com/Future-House/LAB-Bench

11) **BioProBench (2025)** — large-scale protocol QA, step ordering, error correction, protocol generation/reasoning.
    → Paper: https://arxiv.org/abs/2505.07889
    → GitHub: https://github.com/YuyangSunshine/bioprotocolbench

---

## Literature-centric science RAG & agent evals (can aggregate sensitive knowledge)
12) **LitQA / LitQA2** — MCQ that require **full-text** paper retrieval & comprehension; used to evaluate PaperQA2 and other RAG agents.
    → LitQA repo: https://github.com/Future-House/LitQA
    → Paper introducing LitQA2: https://arxiv.org/abs/2409.13740

13) **ScienceAgentBench** — 100+ real scientific tasks (code-executing) to test language **agents** for data-driven discovery.
    → Homepage: https://osu-nlp-group.github.io/ScienceAgentBench/
    → Paper: https://arxiv.org/abs/2410.05080
    → GitHub: https://github.com/OSU-NLP-Group/ScienceAgentBench

---

## Biomedical QA / NLP (broad knowledge & reasoning; **indirectly relevant**)
14) **MultiMedQA** (Med-PaLM/Med-PaLM 2) — aggregation of MedQA (USMLE), MedMCQA, PubMedQA, HealthSearchQA, MMLU-clinical, etc.
    → Overview: https://sites.research.google/med-palm/
    → Nature paper: https://www.nature.com/articles/s41591-024-03423-7
    • Components:
    – PubMedQA → https://pubmedqa.github.io/
    – MedQA (USMLE) → https://github.com/jind11/MedQA
    – MedMCQA → https://medmcqa.github.io/
    – HealthSearchQA → https://huggingface.co/datasets/katielink/healthsearchqa

15) **BioASQ (Task B, Synergy)** — biomedical QA & IR shared task with expert curation.
    → https://www.bioasq.org/

16) **BLURB** — biomedical language understanding & reasoning benchmark (13 datasets, PubMed-centric).
    → https://microsoft.github.io/BLURB/

17) **BLUE** — earlier biomedical language understanding suite (literature + clinical).
    → Paper: https://arxiv.org/abs/1906.05474
    → GitHub: https://github.com/ncbi-nlp/BLUE_Benchmark

18) **MedNLI** — clinician-authored clinical NLI from MIMIC notes.
    → https://jgc128.github.io/mednli/

19) **MEDIQA (Chat/Sum)** — shared tasks on clinical dialogue summarization & generation (doctor–patient).
    → 2023 site: https://sites.google.com/view/mediqa2023

20) **MedDialog (EN/ZH)** — large doctor–patient conversation datasets used for dialogue/assistant evals.
    → Paper: https://aclanthology.org/2020.emnlp-main.743/

---

## Broad bio knowledge evaluations tracking frontier LLMs (context)
21) **LLMs Outperform Experts on Challenging Biology Benchmarks (2025)** — cross-model study spanning molecular biology, virology & biosecurity questions.
    → https://arxiv.org/abs/2505.06108
