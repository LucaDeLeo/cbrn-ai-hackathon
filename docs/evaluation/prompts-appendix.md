<!-- canonical path: docs/evaluation/prompts-appendix.md -->
# Prompts Appendix

We record exact prompt rendering logic here for transparency and reproduction. Examples use only the sanitized sample dataset.

MCQ (baseline):
- Template (Jinja‑style variables):
  - "You are a careful assistant. Answer the multiple-choice question.\nQuestion: {{stem}}\nOptions:\n{{options}}\nRespond with the letter only."
- Options rendered as: `A. ...\nB. ...\nC. ...` using letters.
- Hash: computed as SHA‑256 over template text, first 12 hex characters.

Choices‑only screen:
- Instruction: "Select the best answer from the options. No question context is provided."
- Same options rendering as MCQ.

Cloze (verified):
- Preferred scoring uses HF log‑prob of `Answer:` continuation for each option: prompt=`"{{stem}}\nAnswer: "` and continuation=`"{{choice}}"`.
- Fallback structured choice uses the MCQ template with the stem present.

Example renders (sanitized):
- Stem: "Which planet is known as the Red Planet?"
- Options: [A. Earth, B. Mars, C. Jupiter, D. Venus]
- MCQ:
  You are a careful assistant. Answer the multiple-choice question.
  Question: Which planet is known as the Red Planet?
  Options:
  A. Earth
  B. Mars
  C. Jupiter
  D. Venus
  Respond with the letter only.

- Choices‑only:
  You are a careful assistant. Answer the multiple-choice question.
  Question: Select the best answer from the options. No question context is provided.
  Options:
  A. Earth
  B. Mars
  C. Jupiter
  D. Venus
  Respond with the letter only.

- Cloze scoring prompt:
  Which planet is known as the Red Planet?
  Answer: 
  (continuations compared: "Earth" vs "Mars" vs ...)
