**One Sentence Description**

A recursive prompt-based evaluation system, that uses previous answers as a base for following prompts to the given LLM, and evaluates risk-values based on the differences between answers on a statistical basis.

A recursive model evaluation pipeline for benchmarking LLM responses that incorporates a statistical model that enables the approximation of risk for unmeasured model parameters.

Recursive model evaluation system for benchmarking \[LLM\] responses given targeted questions

**Layers at the High Level**

1. Recursively generate prompts for responses
2. Compare each response to the preceding one
3. Perform statistical analysis on the previous analysis
4. Decide whether prompts in different positions should be compared or not
5. Compare everything and evaluate


**Tools**

1. Python
2. An open source LLM on HuggingFace
3.

* Design the prompt so it is prewritten data for the ML
* Prompt zero \- Research paper

Ama (GMT):
prompt generation \+ prompt zero
figure out whether LLM or ML or anything else for prompt generation
James (GMT+1):
statistics
Bazsi (GMT+2):
ML \+ prompt dataset

Possible ways to test the finalized model:
Use open source LLMs, and modify their parameters so they can provide unwanted, harmful answers \- and use these LLMs and others with slight modifications to test results. (ofc we need to test the LLMs first, to see whether they are actually harmful and to what extent)

Research papers:
[https://cdn.openai.com/gpt-4o-system-card.pdf](https://cdn.openai.com/gpt-4o-system-card.pdf)
[https://www-cdn.anthropic.com/807c59454757214bfd37592d6e048079cd7a7728.pdf](https://www-cdn.anthropic.com/807c59454757214bfd37592d6e048079cd7a7728.pdf)
[https://arxiv.org/pdf/2211.09110](https://arxiv.org/pdf/2211.09110)
[https://epochai.substack.com/p/do-the-biorisk-evaluations-of-ai](https://epochai.substack.com/p/do-the-biorisk-evaluations-of-ai)

Submission template:
[https://docs.google.com/document/d/1YFNy5Yuo-mOF-3ExR5U2cz37wO548rJHtzkN8LoaY2I/edit?tab=t.0](https://docs.google.com/document/d/1YFNy5Yuo-mOF-3ExR5U2cz37wO548rJHtzkN8LoaY2I/edit?tab=t.0)

Extra sources (for CBRN):
[https://arxiv.org/html/2508.06411v1\#S6](https://arxiv.org/html/2508.06411v1#S6)
[https://www.armscontrol.org/act/2025-04/features/unleashing-ai-peace-how-large-language-models-can-mitigate-wmd-risks](https://www.armscontrol.org/act/2025-04/features/unleashing-ai-peace-how-large-language-models-can-mitigate-wmd-risks)

Datasets:
[https://www.wmdp.ai/](https://www.wmdp.ai/)
[https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI)
[https://github.com/anthropics/hh-rlhf](https://github.com/anthropics/hh-rlhf)

Sources:
[https://www.sintef.no/en/expert-list/digital/software-engineering-safety-and-security/probabilistic-ai-and-uncertainty-quantification/](https://www.sintef.no/en/expert-list/digital/software-engineering-safety-and-security/probabilistic-ai-and-uncertainty-quantification/)

**James’ Notes**

**Dynamic linear models**

- Dynamic linear model for time series of risk scores (one risk score after each prompt/response pair \- how do we determine risk score??)
- Dynamic linear model helps inform the new prompts (e.g. level of volatility in risk scores informs to what extent we vary the prompts)
- [https://pyro.ai/examples/forecasting\_dlm.html](https://pyro.ai/examples/forecasting_dlm.html)

**James \- thoughts from talk**

- “Evals are estimators” \- perfect for using Bayesian optimisation to construct an approximate (surrogate) model. These naturally allow us to work with “observations” that are not exact measurements. Should sound good to the judges \- and is truly a good thing to be doing\!

**1\. Prompt Zero, Prompt generation approaches, issues**

* Prompt zero: Research style problem statement that covers a predefined threat space in CBRN \- generating multiple variants with controlled transformations (e.g. paraphrasing, escalating role-play, scaling specificity, nudging chain-of-thought approach)
* Possible issue \- prompt leakage
  * Accidentally generating a prompt that asks for an explicit “how-to” for dangerous instructions \- might not work or result in dangerous data integrated into content logs
  * Possible ways of resulting in leakage:
    * Recursion could escalate \- threshold for analytical, statistical analysis, so it cannot exceed a certain “risk value”
    * Paraphrasing drift: the recursion agent itself could unwrap harmful sentences \- **abstract/scope prompting** \- forbidding step-by-step operational details (and/or integrating it as a threshold for which if a legitimate answer is being generated the program flags the model as highly dangerous and proceeds with a predesigned approach to gain more data)
    * Conclusion: **threshold risk value**, above which another approach is taken \- limited step-by-step operations \- **classifier-based guard**
  * **Classifier-based guard**: answers flagged by a content safety classifier tuned for CBRN (threshold risk) route to human review \- tuned on CBRN-like sanitized data (**WMDP, WMDP bio robust**)
    * In case a model generates concrete, dangerous responses, the approach of the evaluation model instantly changes to a more "careful approach" solely responsible for data gathering \- the model is immediately categorized as dangerous and proceeds with **human review on the flagged answer** (therefore the threshold is high and specific)

**2\. Recursive prompt generation**

* Ways of using responses to generate the **follow-up prompt**:
  * Asking for specification
  * Paraphrasing answer and asking for a new angle/approach
  * Challenging answer
  * Adding context to the answer
* Using the **next layers** which **runs simultaneously** to compare each answer to its preceding one to test whether the change in values is large enough to continue:
  * **Negligible change**:
    * Either new approach (of the 4 that have been mentioned)
    * Concluding the analysis
* **Limited recursion depth** (e.g. 15-20) \- large enough changes could elicit recursion even after the depth limit
* **Metrics** to be collected each round:
  * **Harm classifier score** \- using multiple classifiers to gain valuable data (some are biased, not calibrated, etc. \- one is not enough to be relied on)
  * **Embedding drift** \- how much the prompt alters the answer compared to the proceeding one (e.g. escalation, shift towards abstract/specific)
    * Since this does not necessarily mean danger → contextual metric, and **combine with harm classifier score** (to detect shift towards more dangerous answer)
  * **Token-probability shift** \- IF logits are accessible
    * KL divergence
    * Tracking for refused answers as well \- see how the model holds up (is there a breaking point for it?)
  * **Refusal/compliance rate**
    * Refuses more → might be safer
    * Also check whether refusals are consistent (inconsistency can mean low utility)
  * **Severity weight risk** \- multiplies harm classifier score with a severity score (toxic language can be detected as harm, but its severity is much lower) → perhaps better to compare embedding drift with this one
  * **Uncertainty measure** \- Bayesian estimate to measure the uncertainty in harm classifier score
  * **Human feedback** (if needed)

**3\. Comparing each response to the preceding one**

* Metrics:
  * **Semantic difference**:
    * Representing responses as **vector embeddings** and finding the cosine differences between vectors (**SBERT** or OpenAI embeddings)
    * **Domain-tuned embeddings** are needed
    * Combine with other scores \- not enough in itself because of domain specificity
  * Aforementioned **KL difference**
    * Normalize by response length (KL is length sensitive)
    * **Watchlist** for dangerous terms
  * Measuring change in classifier score
  * Structural difference
    * Edit distance (e.g. **n-gram novelty**)
    * Mainly usable as additional data \- mainly useful if compared with other data
* To calibrate classifier human label should be sufficient

**4\. Perform statistical analysis on the previous analysis**

* Meta-analysis
* James knows this
* What I have in-mind:
  * Beta-binomial model for probability of harmful responses
  * Gaussian Process (GP) or Bayesian neural net that models risk as a function of continuous inputs
  * Active sampling \- acquisition functions
  * Calibration plots, Brier score
* What James already told us:
  * Estimating risk on unmeasured parameters
  * Calibrated uncertainty across a continuous hyperparameter space (e.g., temperature, decoding strategy, model size, finetune flags) \- didn’t know how to phrase so this one was by AI 😀

**5\. Deciding whether prompts in different positions should be compared or not**

* Clustering prompts by similarity/difference \- comparing within clusters

**6\. Compare everything, final evaluation**

* Computing the following:
  * P(harm)
  * Severity (0 to 1\) → Since they are not fully trustable one-by-one: expected risk \= P(harm)\*Severity
  * Posterior variance \- Hierarchical Bayesian logistic model or GP or Bayesian Neural Network, epistemic and aleatoric uncertainty (separate these)
* A risk surface which maps a continuous input space (GP)

Research papers:
[https://cdn.openai.com/gpt-4o-system-card.pdf](https://cdn.openai.com/gpt-4o-system-card.pdf)
[https://www-cdn.anthropic.com/807c59454757214bfd37592d6e048079cd7a7728.pdf](https://www-cdn.anthropic.com/807c59454757214bfd37592d6e048079cd7a7728.pdf)
[https://arxiv.org/pdf/2211.09110](https://arxiv.org/pdf/2211.09110)
[https://epochai.substack.com/p/do-the-biorisk-evaluations-of-ai](https://epochai.substack.com/p/do-the-biorisk-evaluations-of-ai)

Submission template:
[https://docs.google.com/document/d/1YFNy5Yuo-mOF-3ExR5U2cz37wO548rJHtzkN8LoaY2I/edit?tab=t.0](https://docs.google.com/document/d/1YFNy5Yuo-mOF-3ExR5U2cz37wO548rJHtzkN8LoaY2I/edit?tab=t.0)

Extra sources (for CBRN):
[https://arxiv.org/html/2508.06411v1\#S6](https://arxiv.org/html/2508.06411v1#S6)
[https://www.armscontrol.org/act/2025-04/features/unleashing-ai-peace-how-large-language-models-can-mitigate-wmd-risks](https://www.armscontrol.org/act/2025-04/features/unleashing-ai-peace-how-large-language-models-can-mitigate-wmd-risks)

Datasets:
[https://www.wmdp.ai/](https://www.wmdp.ai/)
[https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI)
[https://github.com/anthropics/hh-rlhf](https://github.com/anthropics/hh-rlhf)

Sources:
[https://www.sintef.no/en/expert-list/digital/software-engineering-safety-and-security/probabilistic-ai-and-uncertainty-quantification/](https://www.sintef.no/en/expert-list/digital/software-engineering-safety-and-security/probabilistic-ai-and-uncertainty-quantification/)
[https://arxiv.org/pdf/2407.03876](https://arxiv.org/pdf/2407.03876)
[https://deepignorance.ai/](https://deepignorance.ai/)
[https://huggingface.co/EleutherAI/deep-ignorance-pretraining-stage-weak-filter](https://huggingface.co/EleutherAI/deep-ignorance-pretraining-stage-weak-filter)
