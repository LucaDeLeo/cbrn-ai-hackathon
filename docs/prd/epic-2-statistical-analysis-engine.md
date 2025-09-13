# Epic 2: Statistical Analysis Engine

**Goal**: Implement transparent, from-scratch statistical methods that reveal benchmark biases without requiring model inference. Every implementation must be readable and mathematically documented.

## Story 2.1: Position Bias Analysis

**As a** researcher,
**I want** to detect position preference patterns,
**so that** I can identify if models systematically favor certain option positions.

**Acceptance Criteria:**
1. Calculate frequency distribution of correct answers across positions A,B,C,D
2. Chi-square test for uniform distribution (implemented from scratch with NumPy)
3. Identify questions where correct answer position is predictive
4. Generate position swap variants (A→D, B→C, etc.) for testing
5. Output includes observed vs expected frequencies with p-values
6. Docstring includes mathematical formula for chi-square calculation
7. Automated correctness checks validate option reindexing and label remapping after permutations (unit test with checksums)

## Story 2.2: Bootstrap Confidence Intervals

**As a** researcher,
**I want** bootstrap confidence intervals for all statistics,
**so that** I can quantify uncertainty in bias measurements.

**Acceptance Criteria:**
1. Bootstrap CI implementation in pure NumPy (<50 lines)
2. Support for any statistic function (mean, median, custom)
3. Configurable iterations (default 10,000) and confidence level (default 95%)
4. BCa (bias-corrected accelerated) variant attempted if time allows
5. Performance target: ~10,000 iterations on 3,000 items in seconds on CPU; provide adaptive iteration mode that early-stops when CI width stabilizes; report wall-clock
6. Unit tests verify against known statistical examples
7. Clear documentation of percentile vs BCa methods

## Story 2.3: Lexical Pattern Detection

**As a** researcher,
**I want** to identify textual patterns in exploitable questions,
**so that** I can understand what makes questions vulnerable.

**Acceptance Criteria:**
1. Detect common phrases in correct vs incorrect answers
2. Length distribution analysis with statistical significance
3. Identify "all of the above" / "none of the above" patterns
4. Technical term density comparison
5. Results include top-10 discriminative patterns with effect sizes
6. No external NLP libraries - pure string operations

## Story 2.4: Statistical Battery Integration

**As a** researcher,
**I want** all statistical tests run as a suite,
**so that** I get comprehensive bias analysis in one pass.

**Acceptance Criteria:**
1. Single function runs all implemented statistical tests
2. Consistent output format across all tests (JSON schema)
3. Summary statistics with traffic-light indicators (green/yellow/red)
4. Parallel execution of independent tests
5. Total runtime <30 seconds for 3,000 questions
6. Option to run specific tests only

## Story 2.5: Heuristic Degradation Measurement

**As a** researcher,
**I want** to quantify heuristic performance degradation post-robustification,
**so that** we can demonstrate reduced artifact-driven inflation.

**Acceptance Criteria:**
1. Compute longest-answer, position bias, and lexical heuristic accuracies on original vs robust subsets
2. Report absolute deltas with 95% bootstrap CIs
3. Include results section “Heuristic Degradation” in reports
4. Runtime <5 seconds for heuristic computations on 3,000 items (CPU)

## Story 2.6: Stratified Bootstrap & CI Width Targeting

**As a** researcher,
**I want** stratified bootstrap and CI width monitoring,
**so that** uncertainty is reliable across imbalanced topic distributions.

**Acceptance Criteria:**
1. Bootstrap supports stratified resampling by topic/difficulty when metadata available
2. Report CI widths for key metrics (flagged rate, score deltas)
3. Adaptive iterations increase until CI width stabilizes or max reached
4. Include stratification details in methodology section

## Story 2.7: Confidence-Aware Metrics & Abstention Analysis

**As a** researcher,
**I want** to analyze model calibration and abstention patterns,
**so that** I can identify and reduce hallucination-prone questions.

**Acceptance Criteria:**
1. Calculate abstention rates across confidence thresholds (t=0, 0.5, 0.75, 0.9)
2. Compute calibration metrics (Brier scores, ECE) for model predictions
3. Identify knowledge gap questions (consistent abstention across models)
4. Detect overconfident questions (high error rates without abstention)
5. Generate calibration plots showing predicted vs actual accuracy bins
6. Pure NumPy implementation of scoring function: score = 1 if correct, 0 if abstained, -t/(1-t) if wrong
7. Report includes abstention rate distributions with bootstrap CIs
8. Integrate with existing statistical battery for unified reporting
9. Runtime <10 seconds for calibration analysis on 3,000 questions
10. Output JSON includes per-threshold metrics and pattern classifications
