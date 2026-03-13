# Augenblick Tasks

**Author:** Trained Models  
**Date:** March 2026

## Introduction

This is the solution file for `TASKS.md`.

## Understanding behind tokenization

Machine learning models and Artificial Intelligence, sophisticated as they might be, are just maths in disguise as per our group's philosophy, or the general truth rather. What we wish to accomplish by stating this, is that the inputs to outputs mappings, are rather "Reals" and not the general textual context.

We define a new term here for further learnings.

**Embeddings:** A representation of a word.

Okay, simple. So what? We understand that words can be assigned some number, in context by providing a map of each word to a specific number or real.

But rather, the general trend that helps us in matter of applications, is another definition that is used in practise where the representation is a real-valued vector that encodes the meaning of the word in such a way that the words that are closer in the vector space are expected to be similar in meaning.

Now, generation of these mappings is rather not that convincingly easy.

Okay you got us, it is Maths again here. We use techniques like neural networks, dimensionality reduction on the word co-occurrence matrix, probabilistic models, explainable knowledge base method, and explicit representation in terms of the context in which words appear.

Reference: [Wikipedia — Word embedding](https://en.wikipedia.org/wiki/Word_embedding)

---

## Understanding of Codebase

### Environment Setup

We observed presence of `pyproject.toml`, so it was pretty easy to run `uv sync`.
It set up all the dependencies properly, and got the CLI working.
Verified via `uv run abctokz --help`.

### Benchmarking

**Bug:** We observed while benchmarking that the results only showed benchmarks for `en` and not `hi`.

**Fix:** We fixed it by changing `eval/benchmark.py`. First, benchmark output was incorrectly reporting only `en` because language was hardcoded to `cfg.languages[0]`.
So we added `BenchmarkRunner._build_language_batches()` to benchmark each tokenizer **per language batch** instead of always using only the first language tag. Also changed `BenchmarkRunner.run()` to suit the added language batching.

---

## Task 1

### Studying the Mantra

So to understand this we wrote a small script altering the predefined example file train_bpe.py, 
The corpus has mixed Hindi English and Marathi Language
Okay so upon running this we get a series of almost 76 tokens
<img width="1014" height="191" alt="image" src="https://github.com/user-attachments/assets/c5a697a9-c322-4ca8-b519-773ff1b8884a" />

- From our understanding of the codebase, when encode() is called we first enter tokenizer.py, inside the AugenblickTokenizer class. The tokenizer runs a sequential pipeline: Normalization → Pre-tokenization → Modeling → Post-processing

- First comes normalization. The tokenizer applies Unicode normalization (NFKC) to standardize the text. Since our input was entirely in Devanagari, the string mostly remained unchanged after this step.
Next is pre-tokenization. The WhitespacePreTokenizer splits the normalized text wherever there are spaces or newline characters, breaking the sentence into word-like segments.

- After that, the text goes to the BPE model. BPE is a subword tokenizer that initially breaks words into smaller units and then merges frequently occurring character pairs to form common subwords. Over time, repeated patterns become their own tokens.
Once the subwords are identified, the tokenizer looks them up in the vocabulary table and converts each one into its corresponding integer ID. This list of IDs is the final encoded output.

- For decoding, the SubwordDecoder reverses the process. It takes the token IDs, maps them back to their subwords using the vocabulary, and reconstructs the text while handling special tokens.

- One thing we noticed during testing was the presence of multiple 0 tokens, which seem to represent unknown tokens. Because of this, the decoded text differed noticeably from the original input, suggesting that parts of the Devanagari text were not present in the trained vocabulary.

---

## Task 2

### Responsibility mapping (module-wise)

After reading the core modules and their import structure, the responsibilities map as follows:

| Responsibility | Primary file/module(s) | Why this split exists |
|---|---|---|
| Training a tokenizer (learn vocabulary/pieces from corpus) | `src/abctokz/trainers/base.py`, `src/abctokz/trainers/wordlevel_trainer.py`, `src/abctokz/trainers/bpe_trainer.py`, `src/abctokz/trainers/unigram_trainer.py`; orchestrated by `Tokenizer.train()` in `src/abctokz/tokenizer.py` | Learning logic remains model-specific; top-level class only wires preprocessing + delegation |
| Encode new text with trained tokenizer | `Tokenizer.encode()` in `src/abctokz/tokenizer.py`, plus `src/abctokz/models/*`, `src/abctokz/normalizers/*`, `src/abctokz/pretokenizers/*`, `src/abctokz/processors/*` | Pipeline concerns are centralized; each stage is replaceable |
| Save/load tokenizer artifact | `Tokenizer.save()` / `Tokenizer.load()` in `src/abctokz/tokenizer.py`; helpers in `src/abctokz/utils/io.py`, `src/abctokz/vocab/serialization.py` | Artifact lifecycle is centralized to avoid duplicating persistence logic in each model |
| Measure tokenizer quality (fertility, UNK rate, round-trip, etc.) | `src/abctokz/eval/metrics.py`, `src/abctokz/eval/intrinsic.py`, `src/abctokz/eval/benchmark.py`, reports in `src/abctokz/eval/reports.py` | Evaluation stack is decoupled from training/inference modules |
| Compare against external tokenizers (HF, SentencePiece) | `src/abctokz/adapters/hf.py`, `src/abctokz/adapters/sentencepiece.py`, used by benchmark flows | Adapter layer isolates external dependency APIs from core architecture |

### Core base classes and how they work together

The most important abstractions in this project are the base classes. They define the architecture more than any single implementation file:

- **`Model`** (`src/abctokz/models/base.py`)  
  Core responsibility: tokenize one pre-token into `(token, id)` pairs, and support `save()` / `load()`.  
  Implementations: `WordLevelModel`, `BPEModel`, `UnigramModel`.

- **`Trainer`** (`src/abctokz/trainers/base.py`)  
  Core responsibility: learn a trained `Model` from corpus text (iterator of strings).  
  Implementations: `WordLevelTrainer`, `BPETrainer`, `UnigramTrainer`.

- **`Normalizer` base** (`src/abctokz/normalizers/base.py`)  
  Core responsibility: canonicalize raw text before segmentation (Unicode/whitespace/script-safe transforms).

- **`PreTokenizer` base** (`src/abctokz/pretokenizers/base.py`)  
  Core responsibility: split normalized text into pre-token units. The model cannot cross these boundaries.

- **`PostProcessor` base** (`src/abctokz/processors/base.py`)  
  Core responsibility: transform encodings after model tokenization (for example, adding BOS/EOS special tokens).

- **`Decoder` base** (`src/abctokz/decoders/base.py`)  
  Core responsibility: reconstruct readable text from token strings/IDs according to model family behavior.

- **`Tokenizer` / `AugenblickTokenizer`** (`src/abctokz/tokenizer.py`)  
  Core responsibility: orchestrate the entire pipeline and expose public API (`encode`, `decode`, `train`, `save`, `load`).

### Runtime workflow (what actually happens)

**Training flow:**
The training process follows a structured pipeline:
- A TokenizerConfig object is created through CLI or configuration files.
- Tokenizer.from_config() constructs the pipeline components, including the normalizer, pre-tokenizer, and decoder.
- Tokenizer.train() creates the appropriate trainer using trainers.build_trainer().
- Each corpus line is first normalized and then pre-tokenized before being passed to the trainer.
- The trainer learns the tokenizer model and produces artifacts such as:
  - vocabulary
  - merge rules
  - subword pieces

**Inference flow (encode/decode):**
1. `encode(text)` applies normalizer.
2. Pre-tokenizer splits into pre-token units.
3. Model tokenizes each pre-token into `(token, id)` pairs.
4. Post-processor optionally injects special tokens.
5. `decode(ids)` maps IDs back to tokens and decoder reconstructs text.

**Evaluation flow:**
1. `BenchmarkRunner` loads tokenizer artifacts and corpus samples.
2. `encode_batch()` produces encodings; `decode()` is used for round-trip checks.
3. `eval.metrics` computes fertility, UNK rate, sequence-length ratio, and throughput summaries.

### How import structure confirms this design

The module imports strongly reflect intended boundaries:

- `tokenizer.py` imports models, trainers, decoders, normalizers, pre-tokenizers, and processors. This indicates it is the **orchestrator** layer.
- `trainers/base.py` imports only the abstract `Model` and iterator types. It avoids CLI and evaluation modules; training abstraction stays focused.
- `eval/metrics.py` is mostly pure functions over `Encoding` data. It avoids trainer/model construction logic.
- `adapters/hf.py` and `adapters/sentencepiece.py` import third-party libraries, but expose the same local encode/decode style interface expected by benchmark code.
- `cli/main.py` composes command groups (`train`, `encode`, `decode`, `inspect`, `benchmark`) but does not implement model math itself.

### One boundary that is especially clean

The cleanest boundary is **Trainer → Model**. The contract in `trainers/base.py` is very clear: train on an iterator of corpus strings and return a trained `Model`. This is satisfying for three reasons:

- extension is straightforward (add new trainer + model, then wire builder),
- training logic is isolated from serving-time encode/decode code,
- deterministic behavior expectations are defined at the right abstraction level.

### One boundary that feels blurry/inconsistent

The blur appears in **artifact reconstruction vs full pipeline abstraction**. The architecture presents tokenizer as a full pipeline (normalizer + pre-tokenizer + model + decoder), but load-time behavior is currently model-centric and can diverge from train-time behavior for some configurations.

**What I would do to improve it:**

- persist full `TokenizerConfig` (not only minimal model metadata),
- reconstruct normalizer/pre-tokenizer/post-processor in `Tokenizer.load()` exactly as done in `Tokenizer.from_config()`,
- add strict regression tests asserting that `encode(text)` before save and after load are identical for the same artifact.

This would make module boundaries fully consistent with the intended architecture and improve reproducibility.

---
## Task 3

### The National Anthem Testimage6

For this task, we used the first stanza of **Jana Gana Mana** in two forms:

- English transliteration
- Devanagari script

We trained a tokenizer and encoded both versions using:

```bash
uv run python task3.py
```

Model used in this run: **BPE**  

![Task 3 Raw results](tasks/task3.png)

### Raw results (abctokz)

| Version | Words | Tokens | Fertility (tokens ÷ words) |
|---|---:|---:|---:|
| Transliteration | 55 | 180 | 3.273 |
| Devanagari | 54 | 123 | 2.278 |

Sample tokenization excerpts:

- Transliteration sample tokens:  
  `[Ja, ##na, G, ##an, ##a, M, ##an, ##a, A, ##dh, ##i, ##na, ##ya, ##ka, Ja, ##ya, He, B, ##ha, ##r, ##at, ##a, B, ##ha, ##g]`

- Devanagari sample tokens:  
  `[जन, गण, मन, अ, ##धि, ##ना, ##यक, जय, हे, भ, ##ार, ##त, भ, ##ाग, ##्य, वि, ##ध, ##ा, ##ता, प, ##ंज, ##ा, ##ब, स, ##ि]`

### Interpretation

In this run, **transliteration produced more tokens** than Devanagari (180 vs 123), and therefore had higher fertility (3.273 vs 2.278).

This difference is not caused by script alone. It comes from a combination of:

1. **Script-level symbol patterns** (how character sequences appear and repeat),
2. **Learned vocabulary coverage** of frequent fragments,
3. **Training corpus composition** (which forms and spellings were frequent),
4. **Model family behavior** (BPE merge strategy in this case).

So the outcome is a joint effect of script + data + tokenizer objective.

### Bonus: external tokenizer comparison (`tiktoken`)

The same two texts were tested with `tiktoken` (`cl100k_base`):

| Version | Words | Tokens | Fertility |
|---|---:|---:|---:|
| Transliteration | 55 | 115 | 2.091 |
| Devanagari | 54 | 268 | 4.963 |

### What this reveals

- `abctokz` BPE (trained on the provided corpus) favored Devanagari more than transliteration for this sample.
- `tiktoken` (general-purpose, externally pretrained) produced the opposite pattern: very high token count for Devanagari.

This reveals a key practical point: **fertility is highly tokenizer-dependent**. A domain/script-aware tokenizer trained on relevant data can be much more token-efficient for that script than a generic external tokenizer.

Report saved at: `outputs/task3/report.json`

---
## Task 6

### Making the tokenizer say `<unk>`

For this task, I created and ran a dedicated experiment script:

```bash
uv run python task6.py
```

![Task 6 Results](tasks/task6.png)

The script trains all 3 model families on an **English-only corpus**, then probes difficult inputs:

- Devanagari on English-only model
- Emoji mixed with English
- Rare long English word
- Currency symbol input

Raw report: `outputs/task6/report.json`

### Measured UNK behavior

| Model | Case | Tokens | UNK count | UNK rate |
|---|---|---:|---:|---:|
| WordLevel | devanagari_on_english_model | 2 | 2 | 1.000 |
| WordLevel | emoji_mixed | 3 | 3 | 1.000 |
| WordLevel | rare_english_word | 1 | 1 | 1.000 |
| WordLevel | currency_symbol | 3 | 2 | 0.667 |
| BPE | devanagari_on_english_model | 10 | 10 | 1.000 |
| BPE | emoji_mixed | 10 | 2 | 0.200 |
| BPE | rare_english_word | 22 | 0 | 0.000 |
| BPE | currency_symbol | 10 | 5 | 0.500 |
| Unigram | devanagari_on_english_model | 10 | 10 | 1.000 |
| Unigram | emoji_mixed | 11 | 11 | 1.000 |
| Unigram | rare_english_word | 27 | 27 | 1.000 |
| Unigram | currency_symbol | 10 | 9 | 0.900 |

Aggregate UNK ratio across all test cases:

- **WordLevel:** 0.8889
- **BPE:** 0.3269
- **Unigram:** 0.9828

### At least two different causes of `<unk>`

#### Cause 1: Closed-vocabulary OOV (WordLevel model limit)

**Trigger:** unseen whole pre-token not present in vocabulary.

- Example: `electroencephalographically` on English-only WordLevel model.
- Behavior: entire word becomes `<unk>` in one shot.
- Why: `WordLevelModel.tokenize()` does direct lookup per pre-token and falls back if missing.

This is a **fundamental model-type limit** of word-level lookup.

#### Cause 2: Unseen script/symbol inventory (BPE/Unigram fallback)

**Trigger:** characters/pieces not represented in learned inventory (e.g., Devanagari/emoji after English-only training).

- Example: `नमस्ते भारत` on English-only BPE/Unigram models.
- BPE: all pieces map to UNK IDs when no matching vocab piece exists for those characters.
- Unigram: Viterbi path falls back to `<unk>` pieces for unknown single-character positions.

This is caused by **training data coverage + subword inventory limits**, not only script itself.

### Was it normalizer, pre-tokenizer, training corpus, or model limit?

In this experiment:

- **Primary cause:** training corpus mismatch (English-only) and resulting vocabulary/piece coverage.
- **Model sensitivity:**
  - WordLevel is highly sensitive to unseen tokens (closed vocab behavior).
  - BPE is more robust to rare English words via character/subword decomposition.
  - Unigram here was fragile because many test symbols/pieces were outside its effective learned set.
- **Normalizer/pre-tokenizer contribution:** they define boundaries and canonical form, but they are not the main root cause in these cases. The core failure driver is coverage + model fallback strategy.

### Which model handles unknowns most gracefully? Which is most fragile?

From measured aggregate UNK ratio:

- **Most graceful:** **BPE** (0.3269)
- **Most fragile (in this setup):** **Unigram** (0.9828)

WordLevel is also fragile for any OOV-heavy scenario (0.8889), but still better than Unigram in this specific English-only training setup.

### One concrete suggestion to reduce UNK rate without retraining

Use a **runtime fallback router**:

1. Encode with primary tokenizer.
2. If input-level UNK rate exceeds a threshold (e.g., 0.2), re-encode with a secondary tokenizer better matched to that script/domain (for example SentencePiece/HF adapter or a script-specialized local tokenizer).

This reduces effective UNK in production without changing existing trained weights/artifacts.

---
## Task 8

### What Does the Normalizer Actually Do?

To investigate the normalization, I wrote a script (`tasks/task8.py`) to run the two input phrases through the pipeline and inspect the outputs.

### Raw Input vs Normalized Output

For the two phrases:
- Sindhi: `"आयो लाल, सभई चायो, झूलेलाल!"`
- Marathi: `"गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!"`

**The raw input and normalized output are perfectly identical.** This is because these strings are already fully decomposed and validly encoded in the standard NFC representation without any uncanonical sequences. No character folding or dropping occurred.

![Task 8 Results](tasks/task8.png)

### NFC vs NFKC Normalization

- **NFC (Canonical Composition):** Recombines base characters and their combining marks (matras, halant, etc.) into their pre-composed canonical forms where possible. It generally preserves formatting characters like Zero-Width Joiners (ZWJ) and Zero-Width Non-Joiners (ZWNJ) unless explicitly told not to.
- **NFKC (Compatibility Decomposition + Composition):** Takes normalization further by replacing "compatibility" characters with their standard equivalents (e.g., stripping stylistic ligatures, fractions). Most importantly, **NFKC typically strips formatting characters like ZWJ (U+200D) and ZWNJ (U+200C)** because they are historically considered optional presentation controls.

**Which does this library use for Devanagari?**
The library explicitly uses **NFC** (via `DevanagariNormalizer`) accompanied with an explicit flag `strip_zero_width=False`, explicitly rejecting NFKC for Devanagari.

**Why does that choice matter?**
In Devanagari (specifically for Hindi, Marathi, and Sindhi), ZWJ and ZWNJ are **not** optional styling markers—they fundamentally change the written representation and phonetic structure. They dictate whether consonants form a conjunct character (e.g. `क्ष`) or stay visually split as a half-consonant (e.g. `क्‍ष`). Stripping these out (as NFKC would) is lossy, changing the literal reading and meaning of the text for these languages.

### Commas, Exclamation Marks, and Spaces

During **pre-tokenization**, the `DevanagariAwarePreTokenizer` processes these elements:

- **Spaces** are treated as delimiters and stripped completely. They split the string into a sequence of isolated words.
- **Punctuation** (commas `,` and exclamation marks `!`) remains **attached to the preceding adjacent word**. For instance, `लाल,` becomes exactly one pre-token `['लाल,']`. 

Why? The `_script_of()` classifier tags punctuation as an `"other"` script. The tokenizer merges any `"other"` script characters with the previous non-other script run (the "devanagari" script). As a result, the pre-tokenizer doesn't split punctuation away from Devanagari words. 

**Why this matters for Hindi, Marathi, and Sindhi:**
If punctuation is grouped with adjacent words rather than being pre-tokenized on its own, the BPE/Unigram model will perceive `लाल` and `लाल,` as distinct sequences. This increases the burden on the tokenization vocabulary, causing data sparsity and increasing the likelihood of generating `<unk>` tokens, rather than properly generalizing the core vocabulary separated from pure punctuation blocks.
## Task 9

### Measuring Phrase Difficulty

Using the same two phrases from Task 8:

- **Sindhi phrase:** `आयो लाल, सभई चायो, झूलेलाल!`
- **Marathi phrase:** `गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!`

I trained tokenizers on a Devanagari-rich corpus and measured fertility using:

```bash
uv run python tasks/task9.py
```

Report saved at: `outputs/task9/report.json`

### BPE fertility by vocabulary size

| Vocab size | Sindhi tokens | Sindhi fertility | Marathi tokens | Marathi fertility | Harder phrase |
|---:|---:|---:|---:|---:|---|
| 100 | 19 | 3.800 | 29 | 4.143 | Marathi |
| 400 | 16 | 3.200 | 25 | 3.571 | Marathi |
| 800 | 16 | 3.200 | 25 | 3.571 | Marathi |

### Unigram fertility by vocabulary size

| Vocab size | Sindhi tokens | Sindhi fertility | Marathi tokens | Marathi fertility | Harder phrase |
|---:|---:|---:|---:|---:|---|
| 100 | 7 | 1.400 | 14 | 2.000 | Marathi |
| 400 | 5 | 1.000 | 7 | 1.000 | Tie |
| 800 | 5 | 1.000 | 7 | 1.000 | Tie |

### Which phrase is harder, and why?

At low vocabulary size (`100`), the **Marathi phrase is harder** for both BPE and Unigram (higher fertility).

Reason (observed behavior + model mechanics):

- the Marathi phrase contains longer and more morphologically complex segments,
- those segments split into more subword pieces when vocabulary is tight,
- punctuation and orthographic complexity increase boundary pressure in low-capacity vocabularies.

As vocabulary increases (`400`, `800`), Unigram catches up and both phrases become similarly easy (fertility tie at `1.0`), while BPE still keeps Marathi slightly harder.

### Does fertility change meaningfully with vocabulary size?

Yes, mainly from `100 -> 400`, and then it plateaus:

- **BPE:** fertility drops for both phrases, then stabilizes from `400` to `800`.
- **Unigram:** strong improvement from `100` to `400`; no further gain at `800`.

This indicates diminishing returns: once frequent units are covered, adding more vocabulary gives little additional compression for these short phrases.

### Takeaway

Task 9 puts numbers behind Task 8 observations:

- phrase difficulty is measurable via fertility,
- vocabulary size strongly affects tokenization efficiency at small capacities,
- with adequate vocabulary, model differences narrow for short high-frequency Devanagari phrases.


## Task 13
Okay so im gonna try to remove the exoctic whitespace normalization from the normalizer and see what happens. 
What im predicting is that a few tokenisations tests are going fail so the inheritly when those exoctic charecters are encountered a white space should show but now that it wont show, ther will be tests that fail

Also i think there will be a direct increase in the number of tokens, especially Unowkn tokens

unchanged 
![alt text](tasks/image-1.png)

modified:
![alt text](tasks/image.png)

SO after running the results have been pretty suprising, only one test failed and that was the normalizer integration test, all the other tokenisation tests passed perfectly, 
Now to me this is suprising is because, the thought process was that, the tokens are gonna start to look all weird and messed up but somehow it didnt

So to conclude from this I can say that hidden in our codebase is a secret **Redundancy** and this has some how made it give us a sane response.

## Task 14

### How difficult is adding a fourth model?

Adding a fourth model family like WordPiece is feasible with moderate effort.
The codebase already has clear abstractions in `models/base.py` and `trainers/base.py`, so the core algorithm can be added cleanly.
Most work is integration and plumbing across CLI, config, serialization, and tests.

### Files to create from scratch

- `src/abctokz/models/wordpiece.py`  
  Implements the `Model` abstract interface (tokenization, vocab access, save/load behavior).
- `src/abctokz/trainers/wordpiece.py`  
  Implements the `Trainer` abstract interface (fit/train pipeline and artifact generation).

### Files to modify

- `src/abctokz/tokenizer.py`  
  Register the new model family in load/save dispatch so artifacts can be reconstructed correctly.
- CLI training command (under `src/abctokz/cli/`)  
  Add `wordpiece` as a valid model choice and wire trainer creation.
- `src/abctokz/config/schemas.py`  
  Extend model-type schema validation to include the new model family.

### Files likely unchanged

- Normalizers (`src/abctokz/normalizers/*`)
- Pre-tokenizers (`src/abctokz/pretokenizers/*`)
- Most evaluation metric code (`src/abctokz/eval/metrics.py`)

### Tests to add (repo-specific layout)

In this repository, model tests are grouped in one file.
So the correct approach is to **extend** the existing test module, not create a new standalone one.

- Modify `tests/unit/test_models.py`
- Add a new class `TestWordPieceModel`, mirroring patterns used by
  `TestBPEModel`, `TestUnigramModel`, and `TestWordLevelModel`

Recommended minimum test cases:

- known token/wordpiece segmentation
- unknown token fallback behavior
- empty input handling
- vocab size/access checks
- save/load round-trip correctness

### Where architecture helps vs. where it resists

The architecture helps by providing clean abstract base classes for `Model` and `Trainer` along with a modular pipeline design.
However, some family registration is explicit (hardcoded dispatch), so extension is not fully plug-in based.

### Biggest obstacle

The single biggest obstacle is **artifact compatibility and class dispatch**:
as training the model is only half the work, the critical part is ensuring
`Tokenizer.load()` can reconstruct the new model reliably from saved metadata.
If this integration is incomplete, CLI encode/decode and benchmarking will fail even if the model logic itself is correct.

---

## Task 15

### Find Something That Breaks

We implemented a new, distinct edge-case test suite in: `tasks/task15.py`

Run command:

```bash
uv run python tasks/task15.py
```

![Task 15 Results](tasks/task15.png)

Report generated at:

`outputs/task15/report.json`

### What breaks (two reliable failures, different from decode-UNK reports)

#### Case 1 — Offset metadata integrity is broken for multi-piece tokenization

Reproduction:

```python
text = "internationalization"
encoding = tokenizer.encode(text)
print(encoding.tokens)
print(encoding.offsets)
```

Observed from run:

- tokens: `['i', '##nt', '##er', '##n', '##at', '##i', '##on', '##al', '##iz', '##at', '##i', '##on']`
- offsets: `[(0, 20), (0, 20), (0, 20), ...]` for every piece
- all offsets identical: `True`

Expected:

- If tokenization emits multiple pieces, offsets should correspond to piece-level spans, not the same full-span range for every token.

Classification:

- **Bug** (incorrect metadata; alignment info is unusable).

Reason:

- In `src/abctokz/tokenizer.py` `encode()`, offsets are appended using pre-token length for every emitted piece, and the piece cursor is not advanced per piece.

---

#### Case 2 — Save/load behavior drift (same model, same text, different encoding)

Reproduction:

```python
before = trained_tokenizer.encode("hello world")
trained_tokenizer.save(path)
loaded = Tokenizer.load(path)
after = loaded.encode("hello world")
```

Observed from run:

- before: `tokens=['hello', 'world'] ids=[5, 13]`
- after: `tokens=['<unk>'] ids=[0]`
- behavior changed: `True`

Expected:

- Encoding should be behaviorally consistent before save and after load for the same trained artifact.

Classification:

- **Bug** (persistence contract violation / reproducibility break).

Reason:

- Current load path restores model/decoder, but preprocessing pipeline state used during training is not fully reconstructed.

### Why this approach is stronger

This method demonstrates two independent breakages in two different contracts:

1. **Metadata contract** (offsets must be meaningful for token alignment),
2. **Persistence contract** (save/load should preserve encode behavior).

So this is not only a decode quirk; it reveals architecture-level consistency issues.

### What could we do

- For alignment-sensitive tasks: do not rely on current per-piece offsets until fixed.
- For production reproducibility: validate a short encode parity check after loading artifacts; if mismatch is detected, route to in-memory trained tokenizer or retrain/load via config-preserving path.

### Code fixes

In `src/abctokz/tokenizer.py`:

1. In `encode()`, compute piece-level offsets using per-piece length and advance the piece cursor each loop.
2. In `load()`, restore full pipeline configuration (normalizer + pre-tokenizer + processor) so post-load behavior matches pre-save behavior.

This restores both token-alignment correctness and artifact reproducibility.

### Post-fix verification (`task15after.py`)

To verify the fixes, we added and ran:

```bash
uv run python tasks/task15after.py
```

![Task 15 after revamp results](tasks/task15after.png)

Post-fix report:

`outputs/task15after/report.json`

Observed post-fix output:

- Offsets check:
  - `pass: True`
  - `offsets_identical: False`
  - sample offsets: `[(0, 1), (1, 3), (3, 5), (5, 6), (6, 8), (8, 9)]`
- Save/load check:
  - `pass: True`
  - `behavior_changed: False`
  - before: `tokens=['hello', 'world'] ids=[5, 13]`
  - after : `tokens=['hello', 'world'] ids=[5, 13]`

### Clear before vs after differences

| Check | Before fix | After fix |
|---|---|---|
| Piece offsets for subword tokenization | All offsets identical full-span (`(0, 20)`) | Piece-level spans (incremental offsets) |
| Save/load encode parity | Drift (`['hello','world'] -> ['<unk>']`) | Stable (`['hello','world']` both before/after) |

Conclusion: both identified Task 15 bugs are fixed and validated with reproducible scripts.


## Task 19

### BPE vs Unigram vs WordLevel: what is actually different?

For this task, all three model families were trained on the **same corpus** and with the **same vocabulary size**. To keep the comparison fair, the same five inputs were encoded with all three models, and the outputs were collected into `outputs/task19/report.json`. The helper script used for verification was `task19.py`. One practical detail is worth noting: the script used the in-memory trained tokenizers for final comparison, because the current load path in the library does not fully restore the preprocessing pipeline for `WordLevel`. The script detects this automatically and reports the source used for analysis.

### How it was verified

The experiment was run using:

```bash
uv run python task19.py
```

The five evaluation inputs were:

```text
hello world
internationalization is nontrivial
नमस्ते दुनिया
प्रौद्योगिकीकरण महत्वपूर्ण है
नमस्ते world 2026
```

The discussion below is based directly on the generated token lists, ID sequences, and vocabulary summaries.

### Side-by-side tokenization behavior

- **Easy English: `hello world`**  
  WordLevel returned `[hello, world]`. Unigram also returned `[hello, world]`. BPE instead split the sentence into smaller reusable fragments: `[h, ##el, ##lo, w, ##or, ##ld]`. This is the simplest demonstration that BPE prefers compositional pieces even when whole-word tokens are available.

- **Complex English: `internationalization is nontrivial`**  
  WordLevel kept the input at word level: `[internationalization, is, nontrivial]`. BPE heavily segmented it into many merge-derived pieces such as `[i, ##nt, ##er, ##n, ##a, ##ti, ##on, ...]`. Unigram landed in between with `[inte, rnationalization, is, nontrivial]`. This makes BPE the most fragmentary model and Unigram the most selective subword model in this experiment.

- **Simple Hindi: `नमस्ते दुनिया`**  
  WordLevel produced `[नमस्ते, दुनिया]`, and Unigram also produced `[नमस्ते, दुनिया]`. BPE split the same phrase into `[न, ##मस, ##्, ##ते, द, ##ु, ##नि, ##य, ##ा]`. So in Hindi too, BPE behaves as a genuine subword segmenter rather than a word memorizer.

- **Complex Hindi: `प्रौद्योगिकीकरण महत्वपूर्ण है`**  
  WordLevel and Unigram both kept the three surface words intact: `[प्रौद्योगिकीकरण, महत्वपूर्ण, है]`. BPE decomposed them into a long sequence of subpieces such as `[प, ##्, ##रौ, ##द, ##्य, ##ोग, ... , है]`. This is the clearest example that BPE treats a morphologically rich word as a composition of reusable fragments rather than as a lexical whole.

- **Mixed script: `नमस्ते world 2026`**  
  WordLevel yielded `[नमस्ते, world, 2026]`. Unigram yielded the same. BPE decomposed each span separately into `[न, ##मस, ##्, ##ते, w, ##or, ##ld, 2, ##02, ##6]`. This mixed-script example is useful because it shows that BPE applies the same merge logic across Devanagari, Latin text, and numerals once preprocessing has isolated the boundaries.

### What dominates each vocabulary?

The generated report summarized the vocabularies as follows:

- **WordLevel:** 38 total entries, 37 whole words, 0 subwords, 0 character-like units, 1 special token.
- **BPE:** 200 total entries, 123 subwords, 75 character-like units, 1 whole word, 1 special token.
- **Unigram:** 200 total entries, 106 subwords, 57 character-like units, 36 whole words, 1 special token.

This is the cleanest high-level difference among the three families. WordLevel spends nearly all of its capacity memorizing observed surface forms. BPE spends nearly all of its capacity on reusable fragments. Unigram sits in between: it still prefers a strong subword inventory, but it also preserves many frequent whole words when that improves the probability of the segmentation.

### Which model would I choose?

- **(a) Low-resource language: Unigram**  
  In low-resource settings, the tokenizer must generalize well to unseen forms. WordLevel is weakest here because it depends too much on exact lexical memorization. BPE is better, but Unigram is usually the safer default because it can preserve whole words when useful while still backing off to smaller pieces when data is sparse.

- **(b) Agglutinative or morphologically rich languages such as Hindi or Finnish: Unigram, with BPE as a close second**  
  The complex Hindi example shows why. BPE handles long forms by fragmenting them aggressively, which is useful, but Unigram gives a better balance between keeping frequent long forms intact and decomposing rare forms when necessary.

- **(c) A task requiring consistent token boundaries across languages: WordLevel**  
  If interpretability and stable word boundaries matter most, WordLevel is the best choice. Its outputs are the easiest to inspect and compare across scripts. The trade-off is weaker robustness to unseen words.

### What does each segmentation strategy reveal about its assumptions?

- **WordLevel assumes words are atomic.** If a word exists in the vocabulary, it should remain intact; if it does not, the model has no internal structure to fall back on.
- **BPE assumes frequent adjacent symbol sequences are useful reusable units.** Language is treated as something that can be assembled from common fragments. This makes BPE deterministic and efficient, but also sometimes overly granular.
- **Unigram assumes that the best tokenization is the most probable sequence of pieces.** This is a softer assumption than BPE. Instead of committing to a single merge history, it chooses among candidate segmentations according to learned likelihoods.

### Final intuition

The main lesson from this experiment is that the three models are not just different algorithms; they express different beliefs about language. **WordLevel** believes words should remain words. **BPE** believes reusable fragments are the right building blocks. **Unigram** believes tokenization should be the most probable explanation from a flexible inventory of pieces. In practice, this means WordLevel gives the cleanest boundaries, BPE gives the strongest compositional behavior, and Unigram gives the best overall balance between memorization and generalization.
```

