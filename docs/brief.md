# Project Brief: RobustCBRN Eval

## Executive Summary

RobustCBRN Eval is a practical benchmark robustification toolkit that implements proven bias‑detection techniques from recent research while introducing carefully validated statistical analyses to maximize evaluation validity within realistic hackathon constraints. Our core pipeline applies three complementary approaches: (1) the Deep Ignorance §D.4‑inspired consensus/shortcut detection to identify a substantial fraction of potentially exploitable questions, (2) verified cloze variants that reduce common multiple‑choice artifacts, and (3) a statistical heuristics battery that quantifies remaining biases. We outline additional techniques (e.g., permutation sensitivity testing and mutual information analysis) with minimal proofs‑of‑concept and a roadmap for future comprehensive bias detection.

The problem remains that CBRN benchmarks can be gamed through multiple exploit vectors, with simple heuristics achieving up to 46% accuracy without understanding. Our pragmatic approach focuses on the highest‑impact debiasing techniques that can be reliably implemented and validated within a weekend sprint, while maintaining extensibility for future enhancement and adhering to safety‑aware release practices.

Our framework targets AI safety teams, benchmark developers, and regulatory bodies who need confidence that measured capabilities reflect actual knowledge rather than statistical artifacts. The key value proposition is a robust, validated toolkit that applies proven debiasing techniques with novel enhancements, generates detailed bias reports with statistical confidence intervals (via bootstrap over items), and outputs multiple benchmark variants that together provide more reliable capability assessment than any single evaluation—paired with a conservative security posture.

## Problem Statement

Current AI safety evaluations for CBRN risks suffer from systematic vulnerabilities that could lead to catastrophic misjudgments about model capabilities. The WMDP benchmark, designed to measure dangerous knowledge in biosecurity, cybersecurity, and chemical security, can be exploited through multiple attack vectors that have nothing to do with actual domain understanding.

**Quantified Technical Failures:**
- **31.81% of WMDP-Bio questions** are answerable without reading the question stem (Deep Ignorance §D.4)
- **46% accuracy** achievable by simply selecting the longest answer option
- **~6.5% error rates** in major benchmarks like MMLU, with 5% performance variations from minor formatting changes
- **Position bias**: Models systematically prefer certain option positions (A vs D)
- **Unknown contamination levels**: No systematic checking for training data overlap
- **Zero reproducibility**: No open-source pipelines for validating these statistics independently

**Impact of the Problem:**
The consequences of these evaluation flaws extend far beyond academic concern. A model that achieves high scores through pattern exploitation rather than genuine knowledge could be:
- Falsely certified as "safe" for deployment in sensitive contexts
- Released with inadequate safeguards, creating immediate proliferation risks
- Integrated into laboratory automation systems with catastrophic potential
- Trusted by policymakers making critical decisions about AI governance

**Institutional Recognition of the Crisis:**
This evaluation crisis is recognized at the highest levels of governance and standards bodies. The NIST AI Risk Management Framework explicitly lists CBRN capabilities as the **first** of 12 GAI-specific risks, noting that "current LLMs provide minimal CBRN assistance beyond search engines" but warning this could change rapidly. The 2024 State of AI Regulatory Landscape report bluntly states that **"no proper safety evaluation tools exist yet"**, while the DHS CBRN AI Report highlights "inconsistent testing approaches" and "limited CBRN expertise access" as critical vulnerabilities. Our toolkit directly addresses these documented gaps with transparent, reproducible, open-source methods.

**Why Existing Solutions Fall Short:**
Current evaluation practices rely on single-pass testing with original benchmarks, assuming high scores indicate high capability. There is no standard practice for:
- **Reproducible validation**: No open‑source pipelines for bias detection
- **Statistical confidence**: No error bars or uncertainty quantification on safety claims
- **Consensus testing**: No systematic use of diverse model architectures for validation
- **Transparent methodology**: Closed evaluations that can't be independently verified
- **Community infrastructure**: No shared tools for benchmark robustification
- **Safety‑aware releases**: Limited guidance on avoiding info‑hazardous outputs and per‑item exploit labels

**Urgency with Responsible Disclosure:**
With AI capabilities advancing rapidly and potential integration into research tools accelerating, the window for establishing robust evaluation standards is closing. Every month that passes with flawed evaluations increases the risk of dangerous capability proliferation. However, identifying these vulnerabilities must be done responsibly—our approach focuses on building **defensive infrastructure** that fixes evaluation flaws without creating new attack vectors. We provide open-source tools for the safety community while carefully managing any sensitive findings through established info-hazard protocols.

The time to build this critical infrastructure is now—as a community resource that enables all stakeholders to validate AI safety claims with confidence, before a preventable failure undermines the entire AI safety enterprise.

## Proposed Solution

RobustCBRN Eval provides a pragmatic, well-scoped pipeline that applies proven robustification techniques to MCQA benchmarks, with careful attention to computational constraints and implementation realities.

**Core Implementation (What We'll Actually Ship):**

**Primary Technique: Deep Ignorance §D.4 Robustification**
We implement the validated choices-only screening method:
- Use 2-3 diverse models (different families: Meta/AllenAI/OpenAI lineages)
- Apply majority voting (2/3 models) rather than unanimous consensus for better sensitivity
- Cache all model outputs to disk for reproducibility and compute efficiency
- Expected removal: 25–35% of questions based on prior empirical findings (dataset‑dependent; validated on a small labeled subset)

**Secondary Validation: Statistical Heuristics Battery**
Simple but revealing tests that require no model inference:
- Longest answer selection (baseline: 46% on WMDP-Bio)
- Position bias analysis (first/last option preference)
- Lexical pattern detection (common correct answer phrases)
- These run in seconds and provide immediate insights

**Alternative Format: Verified Cloze Scoring**
Reduces common MC artifacts through independent scoring (residual biases remain):
- Each answer evaluated separately with length‑normalized log‑probabilities
- Requires one forward pass per answer option (≈4× compute)
- Measures how much MC format inflates scores vs cloze (report deltas with CIs)
- Implementation uses HuggingFace Transformers with logit caching

**What We're NOT Implementing (But Could Extended):**
- Complex mutual information calculations (needs more research)
- Adaptive difficulty systems (requires IRT expertise)
- Cross-lingual support (scope to English initially)
- Real-time API integration (focus on batch processing)

**Concrete Technical Specifications:**

**Input Compatibility:**
- JSON format with standard fields: `question`, `choices` (list), `answer` (index or letter)
- CSV with columns: `question_id`, `question_text`, `option_a` through `option_d`, `correct_answer`
- Maximum 10,000 questions per run (compute budget constraint)

**Computational Strategy:**
```python
# Explicit caching architecture
cache_structure = {
    'model_outputs': 'disk',  # JSON files per model/question
    'embeddings': None,        # Not used in v1
    'logits': 'memory',       # For cloze scoring efficiency
    'batch_size': 32,         # Optimal for memory/speed
    'checkpoint_every': 100   # Save progress incrementally
}
```

**Resource Budget:**
- 2 models × 3,000 questions × 2 passes (choices-only + cloze) = 12,000 forward passes
- At $0.03 per 1,000 tokens ≈ $200-300 for full WMDP-Bio evaluation
- Leaves buffer for testing and failed runs

**Safety Implementation:**
- SHA-256 hashing with per-project salt for question IDs
- No storage of question text for sensitive benchmarks
- Aggregate statistics only in public reports
- Optional local-only mode for maximum security

**Why This Focused Approach Succeeds:**
1. **Proven techniques**: Everything we implement has published validation
2. **Realistic scope**: Can actually complete in a weekend
3. **Concrete specifications**: No hand-waving about capabilities
4. **Honest limitations**: Clear about what we don't handle
5. **Extensible foundation**: Clean architecture for future additions

**Near-term Roadmap (Post-Hackathon):**
- Ensemble diversity metrics for model selection
- Semantic similarity clustering for removed questions
- Integration with lm-evaluation-harness
- Multi-language support starting with Chinese

## Target Users

### Primary User Segment: Academic AI Safety Researchers

**Profile:**
- Organizations: University AI safety labs, independent research institutes (MIRI, CAIS, FAR)
- Team size: 2-5 researchers per project
- Technical expertise: PhD students, postdocs, research engineers
- Budget reality: $500-5K for evaluation infrastructure (grant-funded)
- Decision makers: Principal investigators, thesis advisors

**Current Behaviors and Workflows:**
- Publishing papers that require reproducible evaluation methods
- Using public benchmarks due to resource constraints
- Manually checking for evaluation biases in their experiments
- Sharing code and methods for peer review
- Building reputation through methodological rigor

**CBRN-Specific Pain Points:**
- **Info-hazard anxiety**: Worried about accidentally revealing dangerous patterns
- **Domain expertise gap**: CS researchers lack bio/chem knowledge to validate results
- **Dual-use dilemma**: Uncertain which capabilities are genuinely dangerous
- **Reproducibility requirements**: Papers rejected without verifiable methods
- **Limited compute**: Can't run extensive robustness checks

**Goals:**
- Publish credible research on AI safety evaluation
- Contribute open tools to the safety community
- Build methods that industry will eventually adopt
- Establish academic careers in AI safety

### Secondary User Segment: AI Safety Evaluation Organizations

**Profile:**
- Organizations: METR (formerly ARC Evals), Apollo Research, UK AISI technical team
- Team size: 5-15 technical staff
- Technical expertise: Former Big Tech ML engineers, safety researchers
- Budget: $10K-50K per evaluation project
- Influence: Direct input to regulators and major labs

**Current Behaviors and Workflows:**
- Conducting third-party safety evaluations for AI companies
- Building evaluation suites for specific risk categories
- Writing technical reports that inform policy
- Developing standards that may become regulatory requirements
- Bridging the gap between technical and policy communities

**CBRN-Specific Pain Points:**
- **Credibility pressure**: Their assessments influence major decisions
- **Methodology transparency**: Need to justify every evaluation choice
- **Cross-model consistency**: Must work across different architectures
- **Rapid iteration**: New models released faster than evaluation methods
- **Standards void**: No established CBRN evaluation protocols

**Goals:**
- Create evaluation methods that become industry standards
- Provide defensible assessments to regulators
- Build tools that scale across multiple evaluations
- Maintain independence while being practical

## Goals & Success Metrics

### Activity Metrics (What We Do)
- **Benchmarks Processed**: Successfully run on 3+ major CBRN benchmarks
- **Models Evaluated**: Test with 2-3 diverse model families  
- **Runtime Efficiency**: Complete analysis in <4 hours per 1,000 questions

### Output Metrics (What We Produce)
- **Exploitable Questions Identified**: Flag 25-35% as compromised (based on §D.4 findings)
- **Robust Subset Created**: Output includes 65-75% validated questions
- **Statistical Confidence**: Provide 95% CI for all filtering decisions
- **Alternative Formats**: Generate both robust MCQA and verified cloze variants

### Outcome Metrics (What Changes)
- **Score Reliability**: Reduce evaluation variance from 5-10% to <2%
- **True Capability Revelation**: 15-25% score reduction on robust vs original
- **Adoption Indicators**: 20+ GitHub stars, 5+ forks in first week
- **Research Interest**: 3+ teams request collaboration within month
 - **Safety‑Relevance Signal**: On a small, sanitized, non‑actionable subset, robustified scoring reduces inflated performance more for procedural than conceptual items (report deltas with CIs)

### Validation Metrics (How We Know It Works)
- **Cross‑Method Agreement**: Agreement with a 50‑item human‑labeled calibration subset; target 70–80% with error analysis
- **Heuristic Degradation**: Longest‑answer drops materially from ~46% baseline (report absolute delta with bootstrap CIs)
- **Compute Efficiency**: Publish measured throughput/cost; target <$300 for a 1,000–3,000 item run on A100
- **Reproducibility**: <2% variance across three independent seeded runs on a 1,000‑item stratified sample

### Risk Thresholds (When We've Failed)
- If <20% questions flagged: Method too conservative
- If >40% questions flagged: Method too aggressive
- If variance >5%: Not achieving reproducibility
- If no forks/interest: Misjudged user needs

## MVP Scope

### Core Features (Ambitious, With Fallbacks)

**1. Multi‑Model Consensus/Shortcut Detection (up to 3 models)**
- **Implementation**: Target Llama‑3.2‑7B, Mistral‑7B‑v0.3, and Qwen2.5‑7B on Lambda GPU; fallback to 1–2 models if constrained
- **Consensus methods**: Unanimous/majority/weighted voting; report precision/recall vs 50‑item human‑labeled subset
- **Batch processing**: Evaluate 100+ questions with dynamic batch sizing; report throughput
- **Diversity metrics**: Ensure sufficient model disagreement for robust validation

**2. Verified Cloze Scoring with Direct Logit Access**
- **Direct implementation**: Access log probabilities through HuggingFace Transformers
- **Length normalization**: Fair comparison across different answer lengths
- **Parallel evaluation**: Score all options in a single batch
- **Temperature scaling**: Test robustness on a sample; document residual biases

**3. Comprehensive Statistical Analysis Suite**
- **Basic heuristics**: Longest‑answer, position bias, alphabetical preference
- **Optional patterns**: Answer similarity, option interdependence (time‑permitting)
- **Permutation testing**: 0–1 random reorder per item (or stratified subset)
- **Real‑time statistics**: Progressive updates during processing

**4. Smart Caching & Reproducibility System**
- **Multi-level cache**: Embeddings on disk, logits in memory, results incremental
- **Checkpoint system**: Resume from interruption
- **Deterministic execution**: Fixed seeds, ordered processing
- **Full audit trail**: Log every decision for reproducibility

**5. Robust Split with Balance Preservation**
- **Conservative removal**: Prioritize minimal over‑filtering; document criteria
- **Balance checks**: Lightweight topic balance summaries
- **Strategies**: Aggressive, conservative, and balanced splits (documented)

**6. Safety‑Relevance Validation (Misuse Proxy, Safe)**
- **Design**: Construct a tiny, sanitized set of non‑actionable CBRN items partitioned into conceptual vs procedural categories
- **Purpose**: Demonstrate that robustified scoring suppresses spurious gains more on procedural‑like items
- **Safety**: Exclude any operational steps or specific agent details; release only IDs and aggregate metrics

### Stretch Features (If Ahead of Schedule)

- **Semantic Analysis**: Clustering and visualization of removed questions
- **Contamination Detection**: N-gram matching and perplexity analysis
- **Live Demo System**: Interactive demonstration for judges
- **Advanced Metrics**: Mutual information, embedding-based similarity

### Out of Scope for MVP

- ❌ GUI beyond basic demo (CLI primary interface)
- ❌ Multi-language support (English only)
- ❌ Fine-tuning or model training
- ❌ Integration with external evaluation frameworks
- ❌ Real-time API serving

### MVP Success Criteria

**Technical Success:**
- Process 1,000–3,000 questions depending on hardware; publish measured throughput and cost
- Report bias‑detection agreement against a 50‑item human‑labeled calibration subset (precision/recall, not just accuracy)
- Generate 10+ statistical analyses
- Maintain <2% variance across three seeded runs on a 1,000‑item sample

**Demonstration Success:**
```bash
$ python robustify.py --input wmdp_bio.jsonl --models llama mistral qwen --permute 1 --subset 3000
[████████████████████] 3,000/3,000 questions | measured runtime
Results:
- Exploitable (flags): 29.7% ± CI
- Robust subset: 70.3% of items  
- Longest‑answer: 46% → 28%
- Cloze vs MC gap: 18.3% (± CI)
- Balance checks: summary only (no sensitive content)
```

### Implementation Strategy

**Team Division (4 people):**
- Developer 1: Model infrastructure and GPU optimization
- Developer 2: Detection algorithms and consensus logic
- Developer 3: Cloze scoring and advanced metrics
- Developer 4: Analysis, safety, and visualization

**GPU Resource Allocation:**
- Instance: Lambda Labs gpu_1x_a100_sxm4 (40GB) (target)
- Models: Up to 3×7B models; fallback to 1–2×7B quantized if constrained
- Optimization: Flash Attention 2, torch.compile, fp16/bf16; dynamic batching

**Timeline:**
- Friday: Setup, basic implementation (4h)
- Saturday: Core features, full pipeline (12h)
- Sunday AM: Polish, documentation (6h)
- Sunday PM: Demo, submission (4h)

## Post-MVP Vision

### Phase 2 Features (Months 1-3)

**Enhanced Detection Methods:**
- Implement semantic similarity-based shortcut detection
- Add cross-lingual robustness testing
- Develop adaptive thresholding based on model capabilities
- Create ensemble methods combining 10+ diverse models

**Advanced Analysis:**
- Item Response Theory (IRT) integration for difficulty calibration
- Causal analysis of what makes questions exploitable
- Automated report generation with publication-ready visualizations
- Interactive web dashboard for exploration

**Integration & Scaling:**
- lm-evaluation-harness native integration
- API service for continuous benchmark validation
- Distributed processing across multiple GPUs
- Support for 100K+ question benchmarks

### Long-term Vision (6-12 Months)

**Becoming the Standard:**
- RobustCBRN Eval becomes default pre-processing for safety evaluations
- Major labs integrate our methods into their pipelines
- Regulatory bodies adopt our metrics as requirements
- Academic papers routinely cite our robustification

**Technical Evolution:**
- Self-improving system that learns from each evaluation
- Automated discovery of new bias types
- Real-time evaluation during model training
- Predictive modeling of future exploitation techniques

### Expansion Opportunities

- **Domain Extension**: Apply to non-CBRN safety domains (alignment, truthfulness, toxicity)
- **Evaluation-as-a-Service**: Cloud platform for automated benchmark validation
- **Certification Program**: "RobustCBRN Certified" badge for validated benchmarks
- **Research Consortium**: Multi-institution collaboration on evaluation standards

## Technical Considerations

### Platform Requirements
- **GPU Requirements**: Minimum 24GB VRAM, optimal 40GB+ (A100/H100)
- **Memory**: 64GB RAM for caching and parallel processing
- **Storage**: 500GB for model weights and result caching
- **OS Support**: Linux (Ubuntu 20.04+), CUDA 11.8+
 - **Fallback**: Supports 7B quantized models and reduced subsets for local/limited environments

### Technology Stack
- **Frontend**: Streamlit for demo, CLI for primary interface
- **Backend**: Python 3.10+, PyTorch 2.0+, HuggingFace Transformers
- **Models**: Open-source 7B-13B models, quantized 70B for validation
- **Database**: SQLite for local caching, PostgreSQL for production
- **Infrastructure**: Lambda Labs for hackathon, potential Runpod/Vast.ai scaling

### Architecture Considerations
- **Repository Structure**: Monorepo with clear module separation
- **Service Architecture**: Modular pipeline with pluggable components
- **Integration Requirements**: REST API for future service deployment
- **Security/Compliance**: Hash-based anonymization, no plain text storage of sensitive content; avoid releasing hazardous content or per-item exploit labels

## Constraints & Assumptions

### Constraints
- **Budget**: $400 Lambda Labs credits (sufficient for full implementation)
- **Timeline**: 48-hour hackathon sprint
- **Resources**: 4-person team with complementary skills
- **Technical**: English-only for MVP, MCQA format only

### Key Assumptions
- Team has experience with PyTorch and HuggingFace
- Lambda GPU instance remains available throughout
- WMDP benchmark is accessible for testing
- Deep Ignorance §D.4 findings generalize to other benchmarks
- Models selected are sufficiently diverse for consensus

### Submission Alignment
- **Deliverables**: Report using the sprint template, GitHub repository link, optional 3–5 minute video demo, and Appendices: Security Considerations and Prompts Appendix (exact prompt templates and rendering logic)
- **Reproducibility**: Pin seeds and versions, include exact commands, run hashes, and artifact checksums
- **Release Quality**: Public artifacts must pass the Release Checklist (`docs/release_checklist.md`)
- **Interpretation**: Robustified scores reduce artifact-driven inflation and variance but are not a deployment safety clearance

## Risks & Open Questions

### Key Risks
- **GPU Memory Issues**: Models might not fit with aggressive batching
  - *Mitigation*: Dynamic batch sizing, gradient checkpointing
- **Consensus Threshold Tuning**: Optimal threshold unclear without testing
  - *Mitigation*: Make configurable, test multiple values
- **Time Overrun**: Ambitious scope might not complete
  - *Mitigation*: Core features first, clear fallback plan

### Open Questions
- What's the optimal number of models for consensus?
- How much do removed questions overlap across different benchmarks?
- Can we detect contamination without training data access?
- Will permutation testing reveal additional biases?
- How do we validate without ground truth labels?
  - *Plan*: Create a 50‑item calibration subset for human spot‑check (non‑hazardous conceptual items) and report precision/recall.

## Security Considerations
- **Info‑hazards**: Do not release raw hazardous content or per‑item exploitability labels. Publish only item IDs and aggregate metrics.
- **Redaction**: Reports exclude sensitive strings; code includes redaction helpers and a release checklist.
- **Review**: If unsure about potentially hazardous insights, seek organizer review before publishing; follow sprint guidance.
- **Data handling (two‑tier policy)**: Internal artifacts may include plaintext (access‑controlled) to enable debugging and verification; public artifacts must exclude plaintext question content and per‑item details (aggregate only). Use SHA‑256 hashing with two salts: a private per‑project salt for internal artifacts and a fixed public salt for the sanitized calibration subset.
- **Interpretation**: Robustified scores reduce artifact‑driven inflation but are not sufficient evidence for safe deployment.

### Artifacts Release Policy (Summary)
- Internal (private): raw text, full model outputs/logs, full excluded lists, exact rendered prompts (all access‑controlled).
- Public (release): aggregate metrics with CIs, sanitized exemplar snippets (non‑actionable), IDs only for the sanitized calibration subset (public salt), exact prompt templates and rendering logic (Prompts Appendix), no raw hazardous content.

### Prompts Appendix (Reproducibility)
- Publish all exact prompt templates and rendering logic for choices‑only and cloze formats.
- Store exact rendered prompts internally per item/model; include template hashes and versioning in the public Appendix.

### Areas Needing Further Research
- Theoretical foundations for consensus thresholds
- Cross-benchmark bias transfer patterns
- Optimal model diversity metrics
- Long-term stability of robustified benchmarks

## Next Steps

### Immediate Actions
1. **Set up Lambda GPU instance** with PyTorch environment
2. **Download and prepare models** (Llama, Mistral, Qwen)
3. **Obtain WMDP benchmark** and prepare data loaders
4. **Create GitHub repository** with initial structure
5. **Assign team roles** and set up communication channels

### Development Priorities
1. Get single‑model choices‑only detection working (Hour 1–2)
2. Add consensus logic with multiple models (Hour 3–6); define 50‑item calibration subset
3. Implement statistical heuristics (Hour 5–7); add 0–1 permutation pass
4. Build cloze scoring system on a 500–1,000 item slice (Hour 6–10)
5. Create conservative robust splitting (Hour 9–11); document criteria and balance checks
6. Add safety measures and reporting, including Security Considerations appendix (Hour 10–12)

### Post-Hackathon Plans
- Submit to arXiv as technical report
- Apply for compute grants for extended research
- Reach out to METR and Apollo Research for collaboration
- Present at AI safety workshops and conferences
- Open-source with comprehensive documentation

---

## Summary

RobustCBRN Eval addresses a critical gap in AI safety infrastructure by providing practical, validated tools for benchmark robustification. With our Lambda GPU setup and competent team, we can deliver an ambitious MVP that includes multi-model consensus detection, verified cloze scoring, and comprehensive statistical analysis. 

The project targets academic researchers and safety evaluation organizations who need reproducible, trustworthy assessment methods. By focusing on proven techniques while pushing technical boundaries where feasible, we balance innovation with reliability.

Our success will be measured not just by technical metrics but by real-world adoption—if researchers start using our toolkit for their papers and safety teams integrate our methods, we've succeeded in making AI evaluations more robust and trustworthy.

This project matters because flawed evaluations could lead to catastrophic misjudgments about AI safety. Every percentage point of improved evaluation accuracy could be the difference between catching a dangerous capability and missing it. We're building critical infrastructure for the AI safety ecosystem.
