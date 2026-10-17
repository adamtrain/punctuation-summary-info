# Experiment brief: Where does summary information go when punctuation is removed?

Living document. It holds the design, the literature analysis that motivates it (written to feed the
paper's introduction, related-work and discussion sections), and a dated decision log. Update it as
decisions are made; don't rewrite history in the log.

## Context about me
I'm a software engineer (backend/full stack) who is new to mechanistic interpretability. I'm comfortable
with Python but new to TransformerLens, probing, and activation patching. Please explain
interpretability-specific choices as we go, and flag when a design decision matters for validity.

---

## 1. Background

Language models appear to use punctuation tokens (commas, periods, newlines) as "summary points":
attention heads gather the gist of the preceding segment onto the punctuation token's residual stream,
and later tokens read from there. Tigges et al. call this the **summarization motif**.

## 2. Literature review and positioning (for the paper)

Literature check done 2026-10-17 (web and citation search via Semantic Scholar; not exhaustive).
Facts marked **[verify]** came from secondary renderings (abstract pages, HTML conversions) and must be
re-checked against the PDF before they are cited with numbers.

### 2.1 Gap statement (draft wording)

Prior work shows that (a) punctuation positions carry summary information and are causally important
in some models (Tigges et al.; Chauhan & Geiger; Chauhan et al.), and (b) removing punctuation from the
input hurts downstream accuracy (Razzhigaev et al., LLM-Microscope). No work we found removes
punctuation **from the input** and then asks **where the summary information goes**: whether it moves
to substitute positions, spreads out, or is lost. Existing intervention studies ablate punctuation
*activations* while leaving the token in place, which tests whether the model's existing routing relies
on those positions, not how the model routes information when the positions never exist. Existing
input-removal studies measure only task accuracy. We combine input-level removal with linear probes
(where information is *present*) and activation patching (where it is *used*), across models that
differ in how much they rely on punctuation.

### 2.2 Closest prior work, and how we differ

**Tigges, Hollinsworth, Geiger & Nanda (2023), "Linear Representations of Sentiment in Large Language
Models", arXiv:2310.15154.** Introduces the summarization motif.
- Pythia-2.8b on templated "mood stories": patching the full phrase including commas reduced the logit
  difference by about 75%; patching only the pre-comma phrase (commas left alone), about 38%; patching only
  the two comma positions, about 37%. They froze attention patterns, because otherwise "the model could
  simply avoid attending to the commas". **[verify numbers]**
- **Distance effect:** after inserting irrelevant text after each phrase, the ratio of period importance to
  phrase importance rose from about 0.29 (no extra tokens) to about 1.15 (22 extra tokens). Summary points
  matter more as the source gets farther away. **[verify numbers]**
- SST validation used **only sentences containing at least one comma**. Ablating the sentiment direction
  at commas alone accounted for a large share of the total effect. The abstract says 36% of the 76%
  accuracy loss, but the body's numbers (18% of 38%) don't obviously reconcile with that. **[verify; note the
  inconsistency, don't cite one figure without checking]**
- **Non-punctuation summary sites already appear:** in the GPT-2 ToyMovieReview circuit the summary
  position is the noun "movie", not a comma. In Pythia, heads partly relied on a repeated name and
  "feels". Relevance to us: substitute summary sites exist even in punctuated text, so the relocation
  hypothesis must consider content nouns, not only clause-final words and conjunctions.
- What they did not do: remove commas from the input; study comma-free text at all. Their Limitations
  section asks for "more diverse sentence structures".

**Chauhan & Geiger (2024), "GPT-2 Small Fine-Tuned on Logical Reasoning Summarizes Information on
Punctuation Tokens", NeurIPS 2024 workshop.** GPT-2 small fine-tuned on RuleTaker pools rule and sentence
information on period tokens (interchange interventions). This is the precursor to:

**Chauhan, Chaudhary, Choy, Nellessen & Schoots, "Punctuation(s) and Predicates in Language Models",
Findings of EACL 2026, pp. 5622–5636 (arXiv:2508.14067).** The arXiv title says "Punctuation", the ACL
Anthology title "Punctuations".
- Models: GPT-2 small (full fine-tune), Gemma 2B and DeepSeek 1.3B (LoRA), all fine-tuned on RuleTaker
  for 3-way classification.
- Methods: **zeroing** residual-block outputs at periods and the question mark (necessity: zero only those;
  sufficiency: zero everything else), interchange interventions (nnsight), and whole-layer swaps. **No
  linear probes.**
- Result: periods and the question mark are necessary and sufficient in 5 of 12 GPT-2 layers, 1 of 24 in
  DeepSeek, and none in Gemma. In GPT-2, periods are necessary but not sufficient in layers 0–4.
- They *hypothesize* a flow (first sentence → first period in layers 0–4, period → question mark at
  layer 7); it was not directly tested.
- Caveats for how we cite it: (1) zero-ablation is far off the model's normal input distribution and can
  overstate dependence; (2) models were fine-tuned for one task, Gemma and DeepSeek with LoRA, so "Gemma
  does not rely on punctuation" is a claim about those fine-tuned models under zeroing, not about base Gemma;
  (3) the punctuation tokens stay in the input, so the model's routing is unchanged.
- Their stated limitations: punctuation only (no other tokens), periods and question mark in isolation, token-level
  interventions; they suggest finer-grained and subspace analysis.

**Razzhigaev et al., "LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of
Transformers", Findings of NAACL 2025 (arXiv:2502.15007).** **[verify author list]**
- Removes punctuation, stopwords and articles from the input. One mode uses naive rule-based removal;
  the other has GPT-4o remove tokens only where meaning is preserved, without rephrasing.
- Removal models: Llama-3.2-3B, Mistral-7B-v0.1, Llama-3-8B, Qwen2.5-1.5B; benchmarks MMLU (zero-shot) and
  BABILong-4k. Accuracy drops consistently, even when only "irrelevant" tokens are removed.
- Measures **only downstream accuracy** after removal. Per-token "contextualization" is measured on unmodified
  text, and there is no analysis of whether other tokens take over. This is behavioral evidence for our
  **degradation** hypothesis, but it can't tell degradation apart from relocation that works less well.

**Chen et al., "SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator",
ICML 2025 (arXiv:2412.12094).** Keeping only initial, neighboring and separator tokens in attention
preserves most performance. On Llama-3-8B, KV cache shrinks by over 50% on GSM8K-CoT with negligible loss.
Masking experiments show separators are essential. This is an efficiency paper; it does not analyze
representations. Follow-ups (SentenceKV, Sentence-Anchored Gist Compression arXiv:2511.08128, MMSep) are
also systems work.

### 2.3 Adjacent work (context, framing, methods)

- **Attention sinks and massive activations.** Sun et al. (2024), "Massive Activations in Large
  Language Models" (arXiv:2402.17762): in LLaMA2-7B, a few huge activations appear on the **starting token
  and the first period or newline**, act like fixed biases, and attract attention. "House of Cards"
  (arXiv:2410.01866) reports that Mistral-7B has them on the first delimiter rather than the start. This means
  the *first* punctuation token can be a sink, not a summary point. Relevant to the
  pitfalls below, and an alternative explanation reviewers will raise.
- **Ferrando & Voita (2024), "Information Flow Routes" (arXiv:2403.00824).** Reported (secondary source) that
  in Llama 2-7B the first period's residual stream largely bypasses attention and FFN blocks and plays the role of the BOS
  token. **[unverified; read before citing]**
- **Wang et al. (2023), "Label Words are Anchors", EMNLP 2023 best long paper (arXiv:2305.14160).** In in-context
  learning, label words aggregate demonstration information in shallow layers and deep layers read
  from them. It is the same "aggregate, then read" pattern on a different token class, and their saliency-based
  information-flow metric is a cheap complementary method (see Method 3).
- **"Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities", ACL 2026
  (arXiv:2604.10135).** Inserting delimiter tokens at sentence ends and fine-tuning improves reasoning (up to
  +7.7% GSM8K, +12.5% DROP). Ablations suggest the benefit comes from a consistent structural signal. This is the
  converse of our manipulation and supports the claim that boundary tokens are functionally important.
- **Gong et al., "Punctuation-Steered Representation Fine-Tuning (PuReFT)", ACL 2026 short papers.** Fine-tuning
  representations at punctuation positions improves ReFT. This is indirect evidence that punctuation positions are hubs.
- **Liu, Srinivasan & Penn, "Some Tokens Behave like Magnets", EMNLP 2026 (arXiv:2609.05743).** Function words
  act as "repelling magnets" in early layers, and removing them hurts syntactic tasks. Only the abstract was read; it
  does not mention punctuation removal. Possibly relevant to whether function words like "and" become substitute
  boundaries.
- **Geva et al. (2023), "Dissecting Recall of Factual Associations in Auto-Regressive Language Models"
  (EMNLP 2023).** Attributes of a subject (e.g. a city's state) are enriched onto the subject token in early and middle
  MLP layers. This is relevant to a validity concern (Section 6.1): the property may be readable directly from the
  city token without any summary step. (From prior knowledge; not re-checked in this search.)
- **Methods references:** Zhang & Nanda (2023), "Towards Best Practices of Activation Patching"
  (arXiv:2309.16042); Heimersheim & Nanda (2024), "How to use and interpret activation patching"
  (arXiv:2404.15255); Elazar et al., "Amnesic Probing" (arXiv:2006.00995), showing that information found by a probe may
  go unused. Also the probing-control literature (Hewitt & Liang 2019, control tasks).

### 2.4 Discussion hooks (ideas to return to when writing)
- If GPT-2 relocates but Gemma shows no change, that fits Chauhan et al.'s model differences. But we should
  say whether ours is a different measurement (input removal in base models versus activation zeroing in
  fine-tuned models).
- Tigges et al.'s distance effect predicts that removing punctuation matters little for short gaps and a lot for
  long ones. Our filler-length manipulation tests this directly.
- If the summary moves to sentence-initial capitalized words in the "U-cap" condition but not in "U",
  capitalization is acting as a substitute boundary marker. That would be a clean, reportable result.
- U-cap and K together tell a story about capitals: U-cap asks whether *true* capitals stand in for missing
  punctuation, and K asks whether *false* capitals create boundaries that aren't there. Practical framing:
  speech-to-text and dictation output (e.g. Wispr Flow capitalizing after pauses) is a growing share of
  real model input, so robustness to these artifacts has practical relevance.
- M speaks directly to the mechanism behind the summarization motif (identity versus position), which none
  of the prior work separates.

---

## 3. Research question

When punctuation is removed from the input, what happens to the summary information that would normally
collect on punctuation tokens?

### Hypotheses, with operational definitions
The hypotheses are not mutually exclusive; a model could show a mix, and the result may differ by layer.

1. **Relocation:** the summary moves to substitute boundaries (the last word of a clause, the first word of the
   next clause, conjunctions like "and", or content nouns).
   *Signature:* in U, one or a few non-punctuation roles show probe accuracy and/or patching effect clearly
   above their own P-condition level, approaching the level the punctuation roles had in P.
2. **Diffusion:** no single summary point; information is spread across positions.
   *Signature:* in U, many roles are moderately above baseline, none stands out, and patching any
   single position recovers little while patching all downstream positions together recovers a lot.
3. **Degradation:** information about earlier segments is weakened or lost.
   *Signature:* in U, probe accuracy at downstream roles is lower than in P across the board, and the
   behavioral baseline (unpatched logit difference) is lower in U than in P, beyond what the OOD control
   condition explains.

### Research question 2: spurious capitals (condition K)
Motivation: dictation tools such as Wispr Flow often capitalize the first word after a pause, wherever
it falls in the sentence ("We ate lunch, then Went back"). Capitals normally mark sentence starts and
proper nouns, so a stray capital is a false boundary cue. **Does a spurious mid-sentence capital act as a
pseudo-boundary inside the model, and does it hurt?**
- **Pseudo-boundary:** summary information, or a change in attention, collects at the capitalized word or
  the word just before it, as if a new segment started there.
  *Signature:* probe accuracy or patching effect at `pre_cap_word` / `cap_word` rises above the same
  positions in P.
- **Harmless noise:** the capital raises surprise locally but leaves where and how much information is stored unchanged.
- **Harm:** downstream information about earlier segments is weaker (lower probe accuracy and logit difference
  than P, beyond what the unusualness-matched control C explains).
Scope note: this project answers the *mechanistic* version in small open models. "Does it hurt my prompts to
production assistants?" is a behavioral question about large models and would need its own evaluation
(inject dictation-style capitals into real prompts, measure task accuracy across models). Listed as a
possible follow-up, not part of this project's design.

## 4. Design

### 4.1 Dataset (templated, synthetic)
- Short passages where the **first clause sets a property** and the remaining clauses are neutral filler.
- Example property: location (Colorado vs. Texas), set via a place name in clause 1:
  "We drove into Boulder and parked near the trailhead. The weather was fine. We ate lunch, then went back."
- **Filler must be label-independent** (no "the hot dry air" for Texas). The probe must not be able to
  solve the task from filler tokens alone; check this with a probe trained on filler-only text.
- **Filler length is an experimental variable**, not a constant: k ∈ {1, 2, 4, 8, 16} filler sentences.
  Motivation: Tigges et al.'s distance effect.
- **As built (milestone 2):** the property is **country, France vs Japan** (changed from Colorado vs Texas; see
  decision log). Materials are in `punctsum/materials.py`: 24 property-sentence (s0) templates (12 single-clause,
  12 two-clause with a comma after the city clause; the city is never first and always has ≥2 words after it
  in its clause) and 80 filler sentences (40 single-clause, 40 two-clause; second clauses usually start with
  then/and/but/so).
- **Pairs:** each pair is one s0 template plus k fillers, instantiated with a French and a Japanese city of the
  same token count. That gives label balance by construction, cancels any template-label association, and
  provides the clean/corrupted inputs for patching. Stochastic conditions (C, M, K) use a seed derived from the
  pair id, so both members get identical perturbations.
- **Train/test split** holds out templates (6/24 s0, 20/80 fillers, stratified by clause count) **and cities**
  (about 25% of kept cities per token-count group). A probe can only do well on the test split by reading the
  label, not by memorizing a template or a city.
- **City screen:** knowledge margin ≥ 2.0 on "The city of {CITY} is in the country of" (true minus other
  country logit). Cities containing a label name, or that are common English words (Nice), are excluded.
- Code: `punctsum/dataset.py` (screen, split, pairs, rendering), `punctsum/roles.py` (token labels),
  `scripts/m2_generate.py` (generation and checks).
- Generate thousands of passages from varied templates, with balanced labels.
- The property-setting city and the readout answer tokens must tokenize consistently across conditions
  (checked in milestone 1).

### 4.2 Conditions
- **P, punctuated:** normal text.
- **U, unpunctuated (primary):** punctuation removed, **and sentence-initial words lowercased, with proper
  nouns kept capitalized**. Capitals at sentence starts are themselves a boundary cue, so leaving them
  would make U only partially boundary-free. Lowercasing the whole passage would change how the city name
  tokenizes ("Boulder" vs "boulder"), which changes the property token itself, a confound. Templates know
  which words are sentence-initial, so this is easy to do exactly.
- **U-cap, unpunctuated with capitals kept:** punctuation removed, casing unchanged. Tests whether
  capitalized clause-initial words act as substitute boundaries.
- **C, OOD control (word substitution):** punctuation kept; each eligible word is replaced, with probability
  *r*, by a random neutral word from a fixed pool (`punctsum/pilot.py: NEUTRAL_POOL`, chosen to be unrelated to
  place, climate or region). Eligible means: not the city, not sentence-initial (keeps capital cues), not a
  conjunction (keeps "and"/"then" cues), and single-token, with single-token replacements (keeps token counts
  and positions). Purpose: without a control, "punctuation removal" and "strange text" are confounded.
  - **Unusualness** is measured as mean next-token NLL over **word tokens only**. Punctuation tokens are easy to
    predict, so removing them inflates an all-token average without making the words more surprising.
  - **Three strengths (dose-response), not one match:** *r* is set so C's word-NLL increase over P is about
    0.5×, 1× and 1.5× U's. The test is whether U's internal maps differ from P's *more* than C's do at
    similar or greater unusualness. This holds up even though no single unusualness measure is clearly correct.
  - Rejected alternatives: random capitalization (adds false boundary cues, which is what U-cap and K study),
    character-level noise (shatters tokenization), local word shuffling (changes which word sits before
    punctuation, muddying the "last word before punctuation" role).
- **M, misplaced punctuation (extra condition):** U's text (no true punctuation, sentence-initial words
  lowercased) with the same number of commas and periods as P inserted at random **non-boundary** gaps, never
  next to the city. It separates two explanations of the summarization motif: summaries collect on punctuation
  *because it is punctuation* (token identity), or *because it sits at a boundary* (position). It is
  not an OOD control.
- **K, restart capitals (research question 2):** P with *n* random mid-sentence, lowercase, non-proper words
  capitalized (n ∈ {1, 2, 3}, scaled to passage length in milestone 2). Token counts are deliberately not
  preserved: a dictation tool capitalizes whatever word comes next.
- The **readout suffix** (downstream question) is identical across conditions, punctuation included. Only the
  passage is manipulated, so the readout mechanism stays the same.

### 4.3 Token roles
Every token gets a role so positions can be aligned across passages and conditions when token counts
differ. Planned role set (refine after inspecting tokenization):
- `bos`, `property_token(s)` (the city, which may be several tokens; tag first and last), `clause1_content`
- `clause_k_last_word`, `clause_k_punct` (with **first** punctuation of the passage tagged separately
  from later ones, because of the sink issue), `clause_k_first_word` (next clause), `conjunction` ("and",
  "then", ...), `subject_noun` / `repeated_entity` (Tigges-style content summary sites), `filler`,
  `readout_*` (tokens of the question suffix, especially the final position where the answer is predicted).
- Condition-specific: `misplaced_punct` (M) and the words before and after it; `cap_word` and `pre_cap_word` (K).
- Roles are assigned per word or punctuation piece (`punctsum/conditions.py: Piece`), then mapped to tokens by
  character offsets (`punctsum/model.py: tokenize_pieces`); a token that straddles two pieces raises an error.
  For multi-token words, tag first and last token.
- In U and U-cap the `clause_k_punct` roles don't exist; the neighboring roles still do, which is
  what lets us compare.

### 4.4 Method 0: behavioral baseline
Before any internals: for every condition and filler length, measure the unpatched logit difference
between the correct and incorrect answer. If the model can't do the task in U, patching in U is
uninterpretable. This is also the direct behavioral test of degradation, mirroring LLM-Microscope.

### 4.5 Method 1: linear probes (where is the information present?)
- Cache the residual stream (`resid_post`) at every layer, **only at role positions**, on disk.
- For each (layer, role), train a regularized logistic regression probe to predict the property label from
  that single residual vector.
- Only probe positions **after** the property-setting tokens.
- Evaluate on held-out passages, **split by template** to avoid memorizing templates.
- Controls: a shuffled-label baseline for every probe; a filler-only probe (label leakage); comparison of
  punctuation positions with the last content token before them.
- Output: heatmap of probe accuracy by (layer × role), per condition and filler length.
- **As built (milestone 3, `punctsum/probes.py`, `punctsum/probe_positions.py`):**
  - *Probe:* ridge-regression classifier on standardized residuals (train-set mean/std), regularization chosen per
    target by efficient leave-one-out (RidgeCV, generalized cross-validation). Chosen over logistic regression because
    one matrix solve fits the real labels plus 100 permuted label sets at once, which makes a full permutation null
    affordable for every cell. Checked against logistic regression on sample cells (Section 10.8).
  - *Permutation null:* 100 label permutations per cell, **at the pair level**: each training pair's two labels are
    swapped or kept with probability ½. That keeps every pair balanced, which is the right exchangeability unit
    because pair members share everything except the property. Accuracy is always on the true test labels. A cell is
    significant if its accuracy exceeds the null's 95th percentile.
  - *Rows (positions):* the last token of each word, at structural landmarks: in s0, each property word, the word
    after it, the word before each comma/period, the comma/period itself, and the first word after each comma; the
    same landmarks for the first and last filler sentence; every readout token. Rows that only some templates have
    (e.g. a comma inside s0 for country) use the passages that have them; n is reported per row.
  - *Readout:* continuation (primary, per the decision log).

### 4.6 Method 2: activation patching (where is the information used?)
- Minimal pairs: identical passages except for the property (Colorado vs. Texas city).
- Readout: append the fixed question suffix with a one-token answer; metric = logit difference between the
  two answers, normalized so 0 = corrupted baseline and 1 = clean baseline (Zhang & Nanda 2023).
- Patch the residual stream at one (layer, position) from clean into corrupted; measure recovery. Sweep
  all (layer, role) pairs in both conditions.
- Add a "patch all positions after role X" variant to tell diffusion (spread, many weak paths) apart from
  degradation (information simply absent).
- Output: patching heatmaps; compare where the bright spots sit with vs. without punctuation.

### 4.7 Method 3: attention knockout (replaces the struck saliency method)
*Struck 2026-10-17:* saliency/attention-flow scores (Wang et al. 2023). Attention knockout answers the same
question ("which routes carry the information?") causally rather than correlationally, so saliency would add
nothing.

Attention knockout (as in Geva et al. 2023): set chosen attention scores to −∞ before the softmax, at every
layer, so specific query positions cannot read specific key positions. The rest of the computation is untouched.
Knockouts, each applied to all readout tokens as queries:
- **KO-s0words:** readout cannot attend to the property sentence's *words* (the city and its clause). The label can
  then only reach the readout through s0's punctuation or later tokens that already read the city. This is the
  direct test of whether a summary route exists (risk 6.1).
- **KO-s0all:** readout cannot attend to any s0 token, punctuation included. Information must travel through filler tokens.
- **KO-punctonly (P, M):** readout can attend only to BOS and punctuation tokens (plus readout tokens). This tests
  whether punctuation alone suffices as a conduit, SepLLM-style. In M it asks whether misplaced marks can carry the
  label.
Outcome logic: if KO-s0words leaves P's logit diff largely intact, the model routes the label through later
positions, and comparing how much survives in P vs U (vs C) at each k is a behavioral test of relocation vs
degradation. If KO-s0words collapses P too, this property is solved by direct lookback, and a property that must be
aggregated is needed (risk 6.1, decision log).

### 4.8 Interpretation
- Probe and patching agree on a position → strong evidence of a summary point.
- Probe finds information that patching shows unused → present but not read from there (cf. amnesic probing).
- Compare P vs. U (and U-cap, C) maps against the signatures in Section 3.
- **M:** if information collects on misplaced punctuation, punctuation tokens attract summaries regardless
  of position (token identity). If it collects at the true boundaries (the word before them, as in U) and the
  misplaced marks stay unremarkable, the boundary matters, not the token. A mix suggests both.
- **K:** compare the maps around `cap_word` with the same positions in P, against the RQ2 signatures.

## 5. Models, tools, compute

- Start with **GPT-2 small** (known strong effect), then a mid-size **Pythia** (410M or 1.4B), then a
  small **Gemma** (Gemma-2-2B) for comparison with Chauhan et al.
- Library: **TransformerLens** (fall back to nnsight if a model isn't supported). Probes: scikit-learn
  `LogisticRegression` (L2). Plots: matplotlib.
- Environment: project-local `uv` virtualenv (`.venv`), project-local Hugging Face cache (`.hf_cache`).
  Everything installed is recorded in `INSTALL_LOG.md` with cleanup instructions.

### 5.1 RAM budget (24 GB unified memory, Apple M5 Pro)
On Apple Silicon the GPU (MPS) and CPU share the same 24 GB, so model weights, activations and the OS all
come out of one pool.
- **Weights** (fp32 / bf16): GPT-2 small ≈ 0.5 / 0.25 GB; Pythia-410M ≈ 1.6 / 0.8 GB; Pythia-1.4B ≈ 5.6 / 2.8 GB;
  Gemma-2-2B ≈ 10.5 / 5.2 GB. TransformerLens loads Hugging Face weights and then converts them, so **peak memory
  during loading is roughly 2× the model size**. Gemma-2-2B must be loaded in bf16; in fp32 the peak
  (~20 GB) is too close to the limit. Pythia-2.8B and larger are out of scope unless run in bf16.
- **Activations:** cache only role positions, in float16, written to disk per layer (numpy memmap) and
  trained one layer at a time. For example: GPT-2, 4,000 passages × 13 layers × 20 roles × 768 dims × 2 bytes ≈ 1.6 GB per
  condition, which is fine. Gemma-2-2B (27 layers, d=2304), same counts ≈ 10 GB per condition, which is fine on disk but
  must never be fully loaded into RAM; process layer by layer.
- **Patching sweeps:** run in small batches; the memory cost is one forward pass at a time.
- Gemma models are **gated** on Hugging Face: the user must accept the license on the model page and
  provide an access token before milestone 6.

### 5.2 Scaling plan (agreed 2026-10-17 in a side session with the user; recorded here)
Compute: one cloud GPU (assume a single 80 GB A100/H100).
**Machine (checked read-only 2026-10-17):** Lambda instance, SSH alias `lambda`; NVIDIA H100 PCIe 80 GB (driver 580.105,
CUDA 12.8), 26 CPU cores, 221 GB RAM, Python 3.12, no uv. Persistent network storage at `/lambda/nfs/AT1` (empty, 1.8 PB);
local disk 968 GB free (fast, but lost if the instance is terminated).
**Planned layout:** code, outputs and the Hugging Face model cache on `/lambda/nfs/AT1` (they survive restarts; no repeated
multi-GB downloads); the Python environment on local disk (faster imports; rebuildable from `uv.lock` in minutes). All
remote installs are logged in INSTALL_LOG.md. Note: 221 GB RAM fits fp32 weights up to ~27B on the CPU side, but GPU
memory caps fp32 compute at roughly 12–14B, consistent with the bf16 exception for Gemma-2-27B.
- **Order:** finish GPT-2 small through activation patching first and freeze the method; only then scale.
- **Pythia ladder** (160M, 410M, 1.4B, 2.8B, 6.9B, 12B): same training data and order, only size differs, so trends
  with size are interpretable. Full pipeline (behavior, knockout, probes, patching).
- **One modern 7–9B model** (Llama-3.1-8B or Gemma-2-9B) with full internals.
- **Optional:** behavior + knockout only (forward passes, no patching sweeps) on Gemma-2-27B in native bf16 (≈54 GB fits one
  80 GB GPU; fp32 does not). This is a documented exception to the fp32 rule, justified by the bf16-vs-fp32 check on the
  7–9B model.
- **Out of scope:** 70B-class models (not feasible on one GPU).
- **Precision:** no quantized models, ever (int8/int4/GPTQ/AWQ): quantization struggles most with outlier activations,
  which Sun et al. (2024) found on the first period/newline, exactly the positions this project studies. bf16 is not
  quantization; it is the native release format for e.g. Gemma-2 (and fp16 misbehaves there). Standard: load native
  weights, upcast and compute in **fp32** for all internal analyses. Report a cheap bf16-vs-fp32 check (a few hundred
  passages, logit differences). Rerun the GPT-2 baseline on the cloud GPU too, for hardware consistency.
- **Long-distance condition (large models only):** filler of thousands of tokens, since LLM-Microscope found
  punctuation-removal costs for 7–8B models on long-context tasks (BABILong).
- **Bonus:** a larger model may pass the pre-registered mixed-mood (majority) acceptance test that GPT-2 small failed
  (Section 10.7), giving a stronger aggregation property. Apply the same C1/C2 criteria unchanged.

### 5.3 Pre-registered prior for the scaling runs (user's prior; written before any scaling run)
**User's prior, as relayed from the side session:** "In larger models, the NECESSITY of punctuation fades (the behavioral
cost of removing it shrinks), while the USAGE of punctuation as a summary hub when present persists (e.g. the share of
the margin carried through punctuation under knockout stays substantial)."

**Concrete predictions.** Operationalized by Claude from that prior; the user delegated the thresholds to Claude
(2026-10-17). The **directional** predictions (which way each quantity moves with size) are primary; the numeric thresholds
are a secondary guide, written in advance so results can't be read as "consistent with the prior" after the fact. The
measurements themselves don't depend on these thresholds. Measured exactly as for GPT-2 (Sections 10.5–10.7), on the
Pythia ladder:
- **N1 (necessity fades):** U's behavioral cost relative to P on the country property (Δ vs P, pooled over k, question
  readout; GPT-2 small: −0.45) shrinks with size. Predicted: Δ at 12B is smaller in magnitude than at 160M, with
  non-overlapping 95% CIs, and the trend across the ladder is mostly monotonic. The same comparison at the largest k and in the
  long-distance condition is the strongest test.
- **N2 (necessity, relative to the control):** at every size, U's cost stays larger than the matched control C_mid's (the
  punctuation-specific cost doesn't vanish), but its ratio to P's margin decreases with size.
- **U1 (usage persists):** for consistent mood, P's s0words-knockout retention stays ≥ 0.25 (the C2 threshold) at every size.
- **U2 (punctuation is the hub):** the share of P's mood margin carried by s0's punctuation, (s0words − s0all) / none
  (GPT-2 small: 0.38), stays ≥ 0.15 at every size, and patching s0's punctuation span recovers ≥ 0.10 at its best layer.
- **Falsification:** N1 fails if the cost does not shrink (or grows); U1/U2 fail if retention or the punctuation share drops
  below the thresholds in the larger models. Report all outcomes either way.

## 6. Known pitfalls and validity risks

- **Tokenization:** check how each model tokenizes punctuation and spaces (" ," vs ","; merged newlines;
  "Boulder" vs " Boulder" vs "boulder") before building role labels.
- **Attention sinks and massive activations:** the first token, and in some models the **first period or
  newline**, receive large attention or activations for reasons unrelated to summarization (Sun et al. 2024).
  Tag the first punctuation token separately, and in U check whether sink behavior itself moves to another
  token. That would look like relocation but isn't summary relocation.
- **Probe overfitting:** keep probes linear and regularized; always report shuffled-label baselines.
- **Distribution shift:** unpunctuated text is unusual; the loss-matched control C accounts for this.
- **Readout prompt punctuation:** the question suffix has its own punctuation in all conditions; that is
  deliberate (Section 4.2), but note it when interpreting patching at readout positions.

### 6.1 Validity risk: direct lookback to the property token
In short passages the readout position can attend straight back to the city token, whose residual stream
may already be enriched with "is in Colorado" (Geva et al. 2023). Then the readout needs no summary point,
and patching would show the city token and the readout position as bright with little in between, in
**both** conditions. That isn't a bug: it would itself be a finding (summary points aren't needed for this
property at this distance). But it weakens the comparison. Mitigations: the filler-length manipulation
(direct lookback should weaken with distance), and the "patch all positions after role X" variant. If needed,
switch to a property that is aggregated over several tokens rather than carried by one token (e.g. sentiment
or mood expressed by a multi-word phrase, as in Tigges et al.).

### 6.4 Validity risk: the readout supplies separators (found in milestone 2 review)
The question readout starts with `\n\n` and contains `?` and `:`. In U the newline right after the passage is a
natural separator (SepLLM counts newlines as separators), so U is not punctuation-free from the model's point of view:
the summary could relocate to the readout's newline, and we would misread that. Mitigation: a second,
**separator-free continuation readout** (the passage continues in words into the answer), used alongside the question
readout; and readout newline tokens are treated as explicit candidate relocation sites.

### 6.3 Validity risk: the label is readable from the city's spelling (found in milestone 2)
Probes at the city token reach 100% on held-out cities even at layer 0, so the France/Japan distinction is present
in the token embeddings themselves ("Japanese-looking" subwords). Consequences: (1) downstream probes may succeed
wherever any trace of the city's tokens is copied, not only where a "country" summary is formed, so probe accuracy
may saturate and tell hypotheses apart poorly; (2) a single shuffled-label baseline is unstable (Section 10.4).
Mitigations: lean on patching (which measures use, not presence) for the hypothesis comparison; report probe
*selectivity* against a control task; consider a property not visible in the surface form (e.g. one determined by
a combination of words, or a sentiment-style property as in Tigges et al.) if probe maps saturate in milestone 3.

### 6.2 Can GPT-2 small do the task at all?
GPT-2 small may not reliably map cities to states. Milestone 1 checks this behaviorally before building the
full generator; if it fails, choose a property GPT-2 handles (e.g. a more famous city set, or a different
binary property).

## 7. Milestones
1. Environment setup; load GPT-2 small in TransformerLens; inspect tokenization of sample passages in all
   conditions; check that answer tokens are single tokens; measure per-condition loss (to choose control C);
   quick behavioral check that the model can do the property task.
2. Dataset generator with role labels and filler-length variable; sanity-check examples and label balance;
   filler-only leakage probe.
3. Probe pipeline on P only; reproduce the expected spike at punctuation positions.
4. Probes on U, U-cap, C (three strengths), M and K; compare heatmaps. Re-calibrate C's strengths on the
   generated dataset first (pilot calibration is in Section 10.2).
5. Activation patching pipeline (plus behavioral baseline); run on all conditions.
6. Write up findings, then extend to larger models per Section 5.2 (Pythia ladder, one modern 7–9B model, optional
   Gemma-2-27B behavior + knockout, long-distance condition), against the pre-registered predictions in Section 5.3.

## 8. Working style
- Small, verifiable steps; show intermediate outputs (tokenizations, example passages, first heatmaps)
  before scaling up.
- Reproducible: fixed seeds, saved configs, cached activations on disk.
- Flag results that look suspicious or too clean.

## 9. Decision log
- **2026-10-17:** Literature check found no prior work combining input-level punctuation removal with
  analysis of where the information goes; proceeding. Brief updated with: content nouns as relocation candidates;
  filler length as a variable; first punctuation tagged separately (sink risk); behavioral baseline per
  condition; optional attention-flow method; Chauhan et al. caveats.
- **2026-10-17:** Primary unpunctuated condition (U) also lowercases sentence-initial words but keeps proper
  nouns capitalized; U-cap keeps casing. The control perturbation for C will be chosen by matching loss
  increase, measured in milestone 1. The readout suffix is identical across conditions.
- **2026-10-17:** Environment: project-local uv venv and Hugging Face cache; installs logged in
  `INSTALL_LOG.md`. RAM plan in Section 5.1.
- **2026-10-17:** TransformerLens 4.x removed `HookedTransformer`; we use `TransformerBridge` with
  `enable_compatibility_mode()`, which restores HookedTransformer hook names and weight processing
  (LayerNorm folding, centering). Cite TransformerLens 4.2.0 and say compatibility mode was used.
- **2026-10-17 (after milestone 1):** Proper nouns are kept capitalized in every condition (confirmed:
  lowercasing changes city tokenization, e.g. " Denver" → " den" "ver"). Character-level random casing is
  dropped as a control candidate (it shatters tokenization: 43 vs 25 tokens). Patching pairs must match
  city token counts; the generator filters cities by GPT-2 knowledge (Section 10.1) and pairs by token count.
  **Proposed, awaiting confirmation:** control C = random word capitalization restricted to words whose
  capitalized form has the same token count, with probability tuned so the per-token NLL increase matches
  U's (≈ +1.5 nats/token); report total-passage NLL too.
- **2026-10-17:** *Supersedes the proposal above.* Capitalization rejected as the control: capitals are
  boundary cues, so it would inject the variable under study. Control C is now word substitution at three
  strengths (0.5×, 1×, 1.5× U's unusualness), with unusualness measured as word-token-only NLL (Section 4.2).
  Added condition M (misplaced punctuation, user's choice: identity versus position). Added research question 2
  and condition K (spurious capitals, motivated by dictation tools capitalizing after pauses; user's suggestion).
- **2026-10-17 (milestone 2):** Property changed from Colorado vs Texas to **France vs Japan**. GPT-2 small
  reliably knows (margin ≥ 2) only 5 of 35 Colorado cities, so a probe could memorize city identity and no
  Colorado cities could be held out. Screen of 7 label pairs (`scripts/m2_property_screen.py`,
  `outputs/m2/property_screen.txt`): France vs Japan had the most usable cities (22 per side in matched
  token-count pairs); Germany vs Japan was the runner-up (20). Readout is now "Which country was this in?".
- **2026-10-17 (milestone 2):** Test split also holds out cities, not just templates. Filler pool enlarged
  from 60 to 80 sentences so the 20 test fillers exceed the largest k (16). K uses one spurious capital per
  ~10 words. C strengths provisionally r = 0.05 / 0.12 / 0.19, to be recalibrated on the full dataset.
  `Passage` now separates `proper_nouns` (keep their capital: the city, "I", weekdays) from `property_words`
  (the city only; M avoids placing marks next to it).
- **2026-10-17 (design review after milestone 2):** Standing rule from the user: parts of the design that don't
  measure something useful are struck or replaced (results are never altered); controls and baselines stay.
  Concerns raised: (1) the France/Japan property lives in one token and is readable from its embedding, so the
  readout may simply look back at the city and never need a summary, making the hypotheses hard to tell apart (risk 6.1/6.3);
  (2) the readout's newlines and `?`/`:` give U substitute separators (risk 6.4). Agreed plan, before building the
  probe pipeline: run attention knockout (Method 3) and a separator-free readout on the existing data. If lookback
  dominates, add a second property spread over several words (mood/sentiment as in Tigges et al.), keeping
  France/Japan as the single-token contrast. **Struck:** saliency/attention-flow method (made redundant by knockout).
- **2026-10-17 (after knockout, Section 10.6):** Direct lookback carries 90–95% of the France/Japan signal, so per
  the agreed decision rule a second, aggregated property is needed; France/Japan stays as the lookback-dominated
  contrast. Attention knockout is promoted from a check to a core method (cheap, causal, and it already
  separates P from U). **Proposed, awaiting the user's decision:** (a) add an aggregated property (Section 11.2 sketch),
  screened behaviorally and with knockout before any probing; (b) make the continuation readout primary for
  probes and patching, keeping the question readout for behavior and knockout; (c) drop C_lo from the expensive
  internal analyses (behaviorally every C strength costs ≤ 0.10, so C_mid and C_hi carry the dose-response argument).
- **2026-10-17: approved by the user:** (a) build and screen the mixed-valence mood property, with the
  consistent-valence fallback; (b) the continuation readout is primary for probes and patching (the question readout stays
  for behavior and knockout); (c) C_lo is dropped from internal analyses (it stays in the behavioral table).
  Internal-analysis conditions: P, U, U-cap, M, K, C_mid, C_hi.
- **2026-10-17 (mood screen):** Mood answer tokens are " great"/" terrible", not " good"/" bad" (GPT-2's good/bad
  readout is lopsided). Adjectives restricted to a common strength band so majorities aren't overturned by one strong
  word; milder negatives added to the candidates for that. **Struck: the question readout for mood**, since GPT-2 small
  barely answers "good or bad?" (margins +0.09 to +0.35 vs +0.85 to +2.4 for continuations). Pre-registered acceptance
  criteria written into Section 11.2 before the acceptance run. Observed in the screen and to be checked at scale: U
  scored *higher* than P for mood (e.g. consistent +2.41 vs +2.12).
- **2026-10-17 (mood acceptance, Section 10.7):** Per the pre-registered rule, **consistent-valence mood is the main
  aggregated property**; mixed failed C1 (AUC 0.755 < 0.80). France/Japan stays as the lookback-dominated contrast.
  **Proposed strike, awaiting the user's decision:** drop mixed mood from further internal analyses. It failed C1, and
  its knockout pattern duplicates consistent's, so probing it would cost compute without adding a distinct
  measurement. The acceptance result stays recorded as a finding.
- **2026-10-17 (milestone 3 start):** Mixed mood dropped from internal analyses (approved by the user). Milestone 3
  probes run on consistent mood (main) and country (contrast). Probes are ridge classifiers with a pair-level permutation
  null (Section 4.5). **Proposed strike, awaiting the user's decision: the control-task baseline.** A control task
  (random fixed label per city or adjective) detects probes that succeed by memorizing word identity. That only matters
  when test items reuse training types. Our test split holds out cities and adjectives entirely, so a control task scored
  on it is at chance by construction and measures nothing. The held-out-type split plus the permutation null covers the
  same threat, and the layer-0 column shows what is readable from token identity alone.
- **2026-10-17 (milestone 3, first heatmaps):** Ridge and the retrained-permutation null replaced by logistic probes and a
  pair-level permutation of test labels (Section 10.8: the retrained null's 95th percentile reached 1.00; ridge trailed
  logistic by up to 0.32). Recorded confound: probe accuracy is not comparable across rows with different
  lexical variability; the hypothesis tests compare the same row across conditions. **Proposed, awaiting the user's
  decision:** scale probes to k ∈ {1, 4, 16} × all internal conditions × both properties (cheap: about 20 s per run once
  activations are cached), mainly to document presence across conditions and distances. Since probes saturate, the
  hypotheses will be decided by patching (milestone 5), which should come next rather than more probe variants.
- **2026-10-17: approved by the user:** **control-task baseline struck** (redundant with the held-out-type test split);
  probes scaled to k ∈ {1, 4, 16} × {P, U, U-cap, M, K, C_mid, C_hi} × {mood, country}; then activation patching.
- **2026-10-17 (H100 setup):** Project put under git (private GitHub repo adamtrain/punctuation-summary-info; activation
  caches and environments excluded; files over 10 MB need the user's OK before committing). Device selection is automatic
  (`punctsum.model.default_device`: CUDA, then MPS, then CPU; override with `PUNCTSUM_DEVICE`), and shared helpers move
  inputs to the model's device. H100 environment built from the same `uv.lock` (torch 2.14.1+cu130). **Verification:**
  GPT-2 on the H100 reproduces the Mac CPU reference pair to 4 decimals (no knockout 1.6137 / 2.1727; s0words knockout
  0.9245 / −0.6857). TF32 is off for matrix multiplies (`allow_tf32=False`, precision "highest"), so fp32 means fp32. Speed:
  400 renderings in 0.4 s (2.1 GB GPU memory), ~25× the Mac. Scaling runs still wait for the GPT-2 patching batch to finish
  and the method to be frozen.
- **2026-10-17 (models for scaling):** The user has access to the Llama and Gemma families and logged in to Hugging Face on
  the H100; `punctsum/__init__.py` now keeps reading the login token from its default location (`HF_TOKEN_PATH`) while
  models download into the project's cache. Base (non-instruct) models only. Only safetensors weights are downloaded (never
  pickled `.bin`), pinned to exact revisions (`scripts/m6_models.py`, `outputs/m6/models_manifest.json`): pythia-160m
  50f5173d93, -410m 9879c9b5f8, -1.4b fedc38a16e, -2.8b 2a259cdd96, -6.9b c0e3eee36d, -12b bb1e3e710c (fp16 weights, 47.5 GB
  total); Llama-3.1-8B d04e592bb4 (bf16, 16.1 GB); Gemma-2-9B 33c1930284 (37.0 GB). **Correction to the side-session note:**
  the Gemma-2 base repos ship **fp32** weights (Gemma-2-9B 37 GB, Gemma-2-27B 108.9 GB), not bf16. The bf16 exception for
  27B therefore means casting the fp32 release to bf16 to fit the GPU (54 GB). Gemma-2-27B (revision 938270f527, 108.9 GB)
  was pre-downloaded at the user's request so it's available if we run the optional step; whether to run it is still open.
- **2026-10-17 (TransformerLens support):** all scaling models are in TransformerBridge's registry. Pythia ≤ 6.9B,
  Llama-3.1-8B and Gemma-2-9B are verified there (status 1). Pythia-12B and Gemma-2-27B have status 2: verification was
  *skipped* for size ("exceeds 35/100 GB limit"), not failed. **To do before using them:** verify on the H100 that
  TransformerBridge logits match plain Hugging Face logits. **Risk to check:** loading in fp32 with compatibility mode
  (LayerNorm folding, weight centering) may briefly need about 2× the weights' memory; Pythia-12B in fp32 is 48 GB, so a
  2× peak would exceed 80 GB. If so, process the weights in CPU RAM (221 GB) before moving them to the GPU.
- **2026-10-17 (GPT-2 complete; proposed method freeze for scaling, awaiting the user's decision):** GPT-2 small now has
  behavior, knockout, probes and patching for every condition. Proposed protocol per scaling model:
  1. *Per-model setup:* verify TransformerBridge logits against plain Hugging Face logits; record load-time peak memory; a
     bf16-vs-fp32 check on the 7–9B model; recalibrate C's rate (unusualness is model-specific); re-screen cities and
     adjectives (knowledge differs by model) and rerun the mood acceptance test, mixed variant included.
  2. *Behavior + knockout:* all conditions, both properties, k ∈ {1, 4, 16} plus the long-distance condition. These feed
     predictions N1, N2, U1, U2 directly and are cheap (forward passes only).
  3. *Patching (mood only):* P, U, U-cap, M, K, C_mid; k ∈ {1, 4, 16}; 12 evenly spaced layers per model (same relative
     depths across sizes).
  4. *Probes (mood only):* P and U at k=16, to document presence.
  **Proposed strikes for the scaling runs** (all kept for GPT-2): **C_hi** in internal analyses (identical to C_mid in
  behavior, probes and patching, so it adds no measurement; C_lo and C_hi stay in the cheap behavioral table);
  **country probes and patching** (lookback-dominated in every condition; country stays in behavior + knockout, which
  prediction N1 needs); **most probe configurations** (probes saturate and don't discriminate between hypotheses).
- **2026-10-17 (side session with the user, relayed to this session; no files were changed there):** Scaling decisions
  recorded in Section 5.2: one 80 GB cloud GPU; finish GPT-2 through patching and freeze the method first; Pythia ladder
  (160M–12B) then one modern 7–9B model (Llama-3.1-8B or Gemma-2-9B) with full internals; optional Gemma-2-27B behavior +
  knockout in native bf16 as a documented exception; 70B out of scope. **Never quantize** (outlier activations at
  first period/newline are exactly what we study); load native weights, compute in fp32, report a bf16-vs-fp32 check; rerun
  GPT-2 on the GPU for consistency. Add a long-distance (thousands of tokens) condition for large models. The user's prior
  is pre-registered in Section 5.3 before any scaling run; the concrete thresholds there were written by Claude, and the
  user delegated them to Claude's judgment (directional predictions primary, thresholds a secondary guide). A larger model may pass the mixed-mood acceptance test GPT-2 small failed; same criteria apply.
- **2026-10-17:** Conditions are rendered as word/punctuation pieces carrying metadata, and tokens are mapped
  to pieces by character offsets. Verified: the mapped token ids equal `to_tokens()` on every pilot rendering,
  and the milestone 1 report reproduced exactly after the refactor.

## 10. Results log

### 10.1 Milestone 1: GPT-2 small, setup checks (2026-10-17)
Script: `scripts/m1_explore.py`; full output: `outputs/m1/report.txt`, `outputs/m1/results.json`. Seed 0.
48 passages = 3 templates × 16 cities (8 Colorado, 8 Texas). Small and hand-written: these are sanity
checks, not results.

**Tokenization (GPT-2 BPE).** In these passages, punctuation is always its own token, with no leading space
(`,`, `.`); the next word carries the space (`. The` → `.` + ` The`). So punctuation roles are clean
single positions. Words can split (`trailhead` → ` trail` + `head`), so roles must be assigned through a
word-to-token alignment, not by counting words. Single-token cities: Denver, Boulder, Houston, Dallas,
Austin; the rest are 2–3 tokens. Lowercasing changes city tokenization. Answers ` Colorado` and ` Texas`
are single tokens. The readout suffix tokenizes as `\n`, `\n`, `Question`, `:`, ` Which`, ` state`,
` was`, ` this`, ` in`, `?`, `\n`, `Answer`, `:` (13 tokens). Removing punctuation shortens the example
passage from 25 to 21 tokens, so positions after the first sentence shift, as expected.

**How unusual each condition is (mean next-token NLL, nats).**

| condition | per token | per passage | tokens |
|---|---|---|---|
| P | 3.66 | 92.9 | 25.3 |
| U | 5.14 | 109.7 | 21.3 |
| U-cap | 5.15 | 109.8 | 21.3 |
| word-capitalize 30% | 4.88 | 128.1 | 26.2 |
| word-capitalize 60% | 5.19 | 139.7 | 27.0 |
| char-case flip 10% | 6.33 | 215.9 | 34.0 |
| char-case flip 25% | 6.77 | 288.8 | 42.7 |

Removing punctuation costs ≈ +1.5 nats/token. Part of that is mechanical (punctuation tokens are easy to
predict, so removing them raises the average), but total NLL also rises (+17 nats) despite fewer tokens.
U and U-cap are nearly identical, so sentence-initial capitals hardly matter to GPT-2's surprise.
Word capitalization at ~60% matches U per token but not per passage, and it splits some words
(` Parked` → ` Park` + `ed`), which is why the proposed control restricts it to words that keep their token count.

**Behavior.** GPT-2 small knows the city→state mapping when asked directly ("{CITY} is a city in the state of"):
all 16 cities favor the correct state, weakest for Greeley (+0.38), Vail (+0.75) and Telluride (+1.02).
With passage + readout, the correct state wins in 100% of passages in P (mean logit diff +2.11), U (+1.90)
and U-cap (+1.99); word-capitalize-30% gives +2.07 (98%) and char-case-10% gives +1.43 (90%). Top predictions
after the readout include ` Boulder` itself, which suggests the readout attends to the city token.

**Interpretation and flags.**
- At this passage length (two filler sentences), removing punctuation costs only ≈ 10% of the logit
  difference. So behaviorally there is little **degradation**, and the information still reaches the
  readout somehow: by relocation, diffusion, or direct lookback to the city (risk 6.1). The probe and
  patching milestones are what tell these apart.
- This makes the filler-length manipulation essential, not optional: with short gaps the experiment may find
  no difference simply because the readout doesn't need summaries. Milestone 2 should include much longer
  filler (up to ~16 sentences) and re-run this behavioral check across lengths before building probes.
- For patching, only cities with a clear knowledge margin should be used (e.g. diff ≥ 2 on the knowledge
  prompt), paired by token count. The current single-token set (5 cities) is too small; the generator
  should screen a larger city list automatically.
- Peak memory for this run: 2.7 GB RSS. GPT-2 small is no constraint.

### 10.2 Milestone 1b: control calibration, M and K pilot (2026-10-17)
Script: `scripts/m1b_controls.py`; output: `outputs/m1b/report.txt`, `outputs/m1b/results.json`. Same 48 pilot
passages; C, M and K averaged over 5 random draws per passage (240 renderings each). Unusualness = word-token NLL.
"×U" is the condition's increase over P as a multiple of U's increase. The token column includes BOS.

| cond | word NLL | Δ vs P | ×U | total NLL | tokens | logit diff | acc |
|---|---|---|---|---|---|---|---|
| P | 4.01 | 0 | 0 | 92.9 | 26.3 | +2.11 | 1.00 |
| U | 5.14 | +1.13 | 1.00 | 109.7 | 22.3 | +1.90 | 1.00 |
| U-cap | 5.15 | +1.13 | 1.00 | 109.8 | 22.3 | +1.99 | 1.00 |
| M | 5.67 | +1.66 | 1.47 | 144.6 | 26.3 | +1.91 | 0.96 |
| C r=0.1 | 4.97 | +0.95 | 0.84 | 114.4 | 26.3 | +2.05 | 1.00 |
| C r=0.2 | 5.83 | +1.82 | 1.61 | 134.0 | 26.3 | +1.98 | 0.98 |
| C r=0.3 | 6.67 | +2.66 | 2.35 | 152.8 | 26.3 | +1.90 | 0.98 |
| C r=0.5 | 7.77 | +3.75 | 3.32 | 177.3 | 26.3 | +1.84 | 0.99 |
| C r=1.0 | 9.04 | +5.03 | 4.45 | 205.8 | 26.3 | +1.67 | 0.95 |
| K n=1 | 4.48 | +0.46 | 0.41 | 104.0 | 26.6 | +2.08 | 1.00 |
| K n=2 | 4.86 | +0.85 | 0.75 | 113.3 | 26.7 | +2.06 | 1.00 |
| K n=3 | 5.15 | +1.14 | 1.01 | 121.0 | 26.9 | +2.05 | 0.97 |

(Rows for r = 0.4, 0.6 and 0.8 are in the report and follow the same trend.)

**Findings (pilot; no confidence intervals yet; differences of ~0.1–0.2 logits may be noise):**
- With the word-only measure, U costs +1.13 nats/token (versus +1.48 all-token). About a quarter of the earlier
  figure was the mechanical artifact of removing easy punctuation tokens.
- Substitution is potent: r = 0.1 already reaches 0.84×U. The grid was too coarse to pick the 0.5× level
  (r ≈ 0.05); the three strengths should be roughly r ≈ 0.05, 0.12, 0.19. Final calibration happens on the
  generated dataset, since templates and lengths will differ.
- **Suggestive:** at matched unusualness, the control barely affects the answer (C r=0.1: +2.05 vs P +2.11), while
  U drops to +1.90, a drop C only matches at about 2.35× U's unusualness. That leans toward punctuation removal
  costing more than generic strangeness, but the effect is small and needs error bars before it means anything.
- **M** is the most unusual non-control condition (1.47×U) and performs like U (+1.91). Putting punctuation
  back in the wrong places does not restore P-level behavior.
- **K (RQ2):** three spurious capitals in a ~20-word passage are as surprising to GPT-2 as removing all punctuation
  (K n=3: 1.01×U), yet the answer barely moves (+2.05 vs P +2.11). So, first signal: **surprising to the model but
  not harmful to this recall task**. Caveats: GPT-2 small, a recall task the readout may solve by looking
  straight at the city, short passages. Whether false capitals create internal pseudo-boundaries is the probe and
  patching question.
- Peak memory 2.7 GB; runtime 2.7 min on CPU for ~4,000 forward passes.

### 10.3 Milestone 2: generator sample, France vs Japan (2026-10-17)
Script: `scripts/m2_generate.py` (default: 8 pairs per filler length); output: `outputs/m2/sample_report.txt`,
`outputs/m2/pairs_sample.jsonl`, `outputs/m2/city_screen.jsonl`. 40 pairs = 80 passages; 640 renderings
(80 × 8 conditions).

**Checks passed:** every rendering tokenizes with each token inside one word or punctuation piece; appending
the readout never changes passage tokenization; both members of every pair have identical token counts and
role sequences in every condition (required for patching).

**City screen:** 23/27 French cities kept (dropped Limoges, Reims, Metz, Perpignan) and 27/28 Japanese (dropped
Nara). Japanese margins are larger (mostly +6 to +8) than French (+2 to +6). Test cities: Cannes, Rouen, Nantes,
Chamonix, Bordeaux, Avignon / Hiroshima, Kamakura, Okayama, Chiba, Shizuoka, Yokosuka.

**Passage lengths** (mean tokens incl. BOS and the 13-token readout): k=1 ≈ 38 (P) / 35 (U); k=4 ≈ 60 / 53;
k=16 ≈ 166 / 141. Removing punctuation shortens passages by ~15%.

**Behavior (16 passages per cell; no CIs; treat as a hint):** mean logit diff, true minus other country.

| k | P | U | U-cap | M | K | C_lo | C_mid | C_hi |
|---|---|---|---|---|---|---|---|---|
| 1 | +3.17 | +2.86 | +2.95 | +2.80 | +3.06 | +3.21 | +3.18 | +2.98 |
| 2 | +3.81 | +3.43 | +3.51 | +3.34 | +3.66 | +3.72 | +3.61 | +3.52 |
| 4 | +3.12 | +2.79 | +2.80 | +2.66 | +2.93 | +3.03 | +3.09 | +2.87 |
| 8 | +3.30 | +2.70 | +2.74 | +2.67 | +3.18 | +3.28 | +3.00 | +2.99 |
| 16 | +3.33 | +2.67 | +3.04 | +2.72 | +3.03 | +3.35 | +3.18 | +3.29 |

- P is 100% correct at every k, and its margin does not fall with distance: GPT-2 reaches the city from ~165 tokens
  away. Direct lookback (risk 6.1) remains plausible at all lengths.
- U trails P by ≈ 0.3 at k ≤ 4 and ≈ 0.6–0.7 at k ≥ 8, consistent with a distance effect (Tigges et al.).
- M tracks U closely at every k: misplaced punctuation doesn't recover P's behavior.
- K costs ≈ 0.1–0.3; C (provisional strengths) mostly stays close to P.
- Next: confirm with the full dataset and bootstrap CIs before interpreting any of this.

### 10.4 Milestone 2: full dataset, C calibration, leakage and positive controls (2026-10-17)
**Dataset:** `outputs/m2/pairs.jsonl`, 400 pairs per filler length = 2,000 pairs (1,500 train / 500 test) =
4,000 passages per condition. All 32,000 renderings pass the tokenization and pair-consistency checks
(`outputs/m2/report.txt`). City token counts: 3 tokens 49%, 2 tokens 32%, 1 token 19%. Mean length with readout:
37 tokens (k=1, P) to 167 (k=16, P).

**C calibration** (`scripts/m2_calibrate_c.py`, `outputs/m2/calibration.txt`; 200 training pairs, 4,800 renderings):
U raises word-NLL by **+0.57 nats/word** over P (k=1 +0.60 … k=16 +0.45), about half the pilot's +1.13. Likely because
missing boundaries mostly surprise the model at the next word, and these sentences are longer (fewer boundaries per
word). Substitution unusualness rises steeply and roughly linearly with r (r=0.06 ≈ 1.03×U, r=0.26 ≈ 4.1×U).
Chosen rates: **C_lo r=0.026 (0.5×U), C_mid r=0.057 (1×U), C_hi r=0.079 (1.5×U)**.
*Caveat for the paper:* at these rates short passages often get 0–1 substitutions, so C is "rare but strong" where U
is "frequent but mild". Matching is on average unusualness; the three strengths guard against relying on one match.

**Leakage and controls** (`scripts/m2_leakage.py`, `outputs/m2/leakage_report.txt`; k=4 pairs, 300 train / 100 test):
- *Filler leakage is impossible by construction* (both members of a pair share identical filler, drawn independently
  of the label), so this checks the probe pipeline. **Negative control:** probes on filler-only text (one random
  member per pair) score 0.43–0.57 at every layer, all inside the 95% chance band [0.40, 0.60] for n=100.
  So with 100 test items, ±10 points from 50% is noise; milestone 3 heatmaps need that band drawn on them.
- **Positive control at the city's last token (P):** 1.00 test accuracy on held-out templates *and cities* at **every
  layer, including layer 0 (embeddings)**. French and Japanese city names are linearly separable from their tokens
  alone (spelling and subwords), before any context. See risk 6.3.
- **At the final readout position (P):** test accuracy rises with depth, 0.50 (L0) → 0.69 (L1) → 0.84 (L3) →
  0.90 (L6) → 0.96 (L7) → 1.00 (L10–12). The answer is assembled gradually through the network.
- **Single-shuffle baselines are unreliable here:** one shuffled-label probe scored 0.21–0.53 at the city token
  (far below chance) and up to 0.77 at the readout. Cause: the shuffled labels correlated −0.013 with the true ones
  (1,000-permutation 95% range ±0.08), and because the true signal is so strong, the probe amplifies even that tiny
  overlap. **Milestone 3 must use a permutation null distribution (many shuffles) and a control-task probe**
  (Hewitt & Liang 2019: each city gets a random fixed label) instead of a single shuffle.

### 10.5 Milestone 2: full behavioral baseline with CIs (2026-10-17)
Script: `scripts/m2_behavior.py`; output: `outputs/m2/behavior_report.txt`, `behavior_summary.json`, per-rendering
`behavior.jsonl`. 2,000 pairs × 2 labels × 8 conditions = 32,000 renderings, question readout. 95% CIs from 2,000
bootstrap resamples of pairs; Δ is paired by pair. Runtime 11 min on CPU, peak 7.6 GB.

Δ vs P (logit diff; P itself is +3.85 at k=1, falling to +2.92 at k=16):

| k | U | U-cap | M | K | C_lo | C_mid | C_hi |
|---|---|---|---|---|---|---|---|
| 1 | −0.35 [−0.36, −0.33] | −0.22 | −0.39 | −0.11 | −0.02 | −0.04 | −0.08 |
| 2 | −0.36 | −0.29 | −0.42 | −0.12 | −0.02 | −0.05 | −0.07 |
| 4 | −0.40 | −0.41 | −0.48 | −0.16 | −0.03 | −0.07 | −0.07 |
| 8 | −0.53 | −0.51 | −0.58 | −0.22 | −0.02 | −0.05 | −0.10 |
| 16 | −0.60 [−0.62, −0.57] | −0.31 | −0.59 | −0.24 | −0.02 | −0.06 | −0.08 |
| all | −0.45 [−0.46, −0.44] | −0.34 | −0.49 | −0.17 | −0.02 | −0.06 | −0.08 |

All Δ CIs are narrow (about ±0.02–0.03) and exclude 0. Accuracy is ≥ 98.6% in every cell, so the logit diff is the
sensitive measure. Measured unusualness (×U) per condition: U-cap 0.6–0.9, M 2.0–2.8, K 1.2–1.5, C_lo ≈ 0.45,
C_mid ≈ 1.0, C_hi ≈ 1.2–1.6, so the calibration holds across k.

**Findings (behavioral; this says nothing yet about *where* information sits):**
- **Punctuation-specific cost.** At matched unusualness, U costs 7–10× more than the control (−0.45 vs −0.06
  overall), and even C_hi (more unusual than U) costs only −0.08. The cost of removing punctuation is not explained by
  the text being strange.
- **Distance effect.** U's cost grows from −0.35 (k=1) to −0.60 (k=16) with non-overlapping CIs, matching the
  prediction from Tigges et al. that summary points matter more as the source gets farther away. P itself also declines
  with distance (+3.85 → +2.92), so direct lookback weakens with distance too.
- **Capitals as a partial substitute boundary.** U-cap matches U at k=4–8 but recovers about half the loss at k=16 (−0.31
  vs −0.60) and some at k=1–2. Sentence-initial capitals seem to act as boundary cues when punctuation is absent.
- **Misplaced punctuation doesn't help** (M ≈ U at every k, slightly worse at short k), although the punctuation
  tokens are present. Behaviorally, being at a true boundary matters, not the token itself (to be checked internally).
- **RQ2, stray capitals (K):** −0.11 (k=1) to −0.24 (k=16), 2–3× the cost of control noise of similar unusualness,
  and growing with distance. That fits false capitals acting as (misleading) boundary cues. It is small in absolute
  terms (≈5% of the margin, no accuracy change) on this task and model.

### 10.6 Milestone 2 review: attention knockout × readout (2026-10-17)
Script: `scripts/m2_knockout.py`; output: `outputs/m2/knockout_report.txt`, `knockout_summary.json`, `knockout.jsonl`.
500 pairs (100 per k), 6 conditions, 2 readouts, 4 knockouts; pair-level logit diff with 95% bootstrap CIs. Runtime
14.5 min, peak 7.9 GB. Readout choice for the continuation readout: `scripts/m2_readout_screen.py`,
`outputs/m2/readout_screen.txt` (" and all of this was in the country of": P +4.83 / U +4.59 on 100 pairs, versus the
question readout's +3.36 / +2.93).

Pooled over k (pair-level logit diff):

| readout | cond | none | s0words | s0all | punctonly |
|---|---|---|---|---|---|
| question | P | +3.49 | +0.16 | +0.12 | +0.20 |
| question | U | +3.08 | +0.07 | +0.07 | — |
| question | U-cap | +3.19 | +0.09 | +0.09 | — |
| question | M | +3.03 | +0.14 | +0.12 | +0.19 |
| question | K | +3.34 | +0.20 | +0.16 | +0.21 |
| question | C_mid | +3.44 | +0.16 | +0.13 | +0.22 |
| continuation | P | +5.11 | +0.49 | +0.38 | +0.38 |
| continuation | U | +4.91 | +0.26 | +0.26 | — |
| continuation | U-cap | +5.04 | +0.34 | +0.34 | — |
| continuation | M | +4.53 | +0.54 | +0.46 | +0.45 |
| continuation | K | +4.93 | +0.56 | +0.46 | +0.40 |
| continuation | C_mid | +4.99 | +0.50 | +0.39 | +0.41 |

CIs are about ±0.01–0.03 for knockout cells; pair accuracy stays ≈1.00 even under knockout, because the small
residual margins are highly consistent. Per-k tables are in the report.

**Findings:**
- **Direct lookback dominates (risk 6.1 confirmed).** Blocking the readout from the property sentence's words
  removes ~95% of P's margin with the question readout (+3.49 → +0.16) and ~90% with the continuation readout
  (+5.11 → +0.49). Only 5–10% of the signal reaches the readout through later positions. Under the decision rule agreed in
  the design review, France/Japan alone cannot carry the summary question: a property that must be aggregated is needed.
- **The residual route is real and punctuation-sensitive.** What survives s0words is about twice as large in P as in U
  (+0.16 vs +0.07 question; +0.49 vs +0.26 continuation), with U-cap in between and C_mid ≈ P. So without punctuation
  less label information is available through later positions, beyond what unusualness explains.
- **The s0 period carries part of it:** in P, also blocking s0's punctuation (s0all vs s0words) removes ~25% of the residual
  (+0.16 → +0.12; +0.49 → +0.38). In U the two knockouts are identical by construction (sanity check holds).
- **Punctuation alone is a sufficient conduit for the residual (SepLLM-like).** With only BOS and punctuation visible
  (punctonly), P keeps +0.20 (question), as much as or more than s0words, which also leaves the filler words visible.
  Plausibly, filler words dilute attention away from the punctuation. punctonly grows with k (more punctuation tokens):
  question +0.16 (k=1) → +0.22 (k=16); continuation +0.27 → +0.54.
- **Misplaced punctuation also works as a conduit:** M's punctonly matches or exceeds P's (+0.19 vs +0.20 question;
  +0.45 vs +0.38 continuation). Internally, punctuation tokens collect label information even at wrong positions. Yet
  behaviorally (no knockout) M ≈ U. Token identity seems to matter for *collecting* information, position for *using*
  it; probes and patching should test this.
- **The U behavioral deficit is larger than the whole residual route.** Without knockout, U trails P by 0.41
  (question), more than P's entire non-lookback route (0.16). So most of U's cost must come through the direct-lookback
  path being less effective without punctuation (e.g. weaker enrichment of the city or less precise retrieval).
  *Caveat:* logits are not additive across routes once the main route is knocked out, so this decomposition is
  suggestive, not exact.
- **Readout format matters for U's deficit:** with the continuation readout U equals P at k=1–2 and falls behind
  only at k ≥ 4 (−0.11, −0.33, −0.54 at k=4, 8, 16). The distance effect is cleaner without the question's separators.

### 10.7 Milestone 2b: mood acceptance test (2026-10-17)
Scripts: `scripts/m2b_mood_screen.py` (re-run with the final adjective band), `scripts/m2b_mood_accept.py`. Output:
`outputs/m2b/accept_report.txt` (per-k tables), `accept_summary.json`, `accept.jsonl`, `pairs_{mixed,consistent}.jsonl`.
500 pairs per variant (100 per k), continuation readout " and overall the whole experience was", answers
" great"/" terrible". Runtime 16 min, peak 7.2 GB.

**Verdict (pre-registered, Section 11.2):**
- mixed: pair acc 0.998, **AUC 0.755 → C1 FAIL**; s0words retains 0.54 [0.51, 0.57] → C2 pass.
- consistent: pair acc 1.000, AUC 0.967 → C1 pass; s0words retains 0.51 [0.49, 0.53] → C2 pass.
→ **Consistent mood is the main aggregated property.** The mixed variant missed C1 narrowly; the bar was not moved.
GPT-2 small does not reliably compute a majority valence across passages, though within a pair it always moves in the
right direction.

Consistent mood, pooled over k (pair-level logit diff, great − terrible for the true label; retention = s0words / none):

| cond | none | s0words | s0all | punctonly | s0words retention |
|---|---|---|---|---|---|
| P | +1.67 | +0.85 | +0.21 | +0.43 | 0.51 [0.49, 0.53] |
| U | +1.80 | +0.16 | +0.16 | — | 0.09 [0.07, 0.10] |
| U-cap | +1.57 | +0.20 | +0.20 | — | 0.13 [0.12, 0.14] |
| M | +1.77 | +0.60 | +0.29 | +0.44 | 0.34 [0.32, 0.35] |
| K | +1.67 | +0.81 | +0.23 | +0.40 | 0.49 [0.47, 0.50] |
| C_mid | +1.60 | +0.82 | +0.22 | +0.42 | 0.51 [0.49, 0.52] |

P's retention falls with distance: 0.62 (k=1), 0.55, 0.49, 0.44, 0.35 (k=16). U stays at 0.05–0.11 at every k. The
mixed variant shows the same pattern (P 0.54, U 0.07, U-cap 0.13, M 0.30, K 0.50, C_mid 0.54).

**Findings:**
- **A real summary route, carried by the property sentence's own punctuation.** In P, half the mood margin reaches
  the readout without the readout ever attending to the adjectives (s0words: +0.85 of +1.67). Also blocking s0's two
  commas and period (s0all) drops it to +0.21. So s0's three punctuation tokens carry about 0.64 of the 1.67 margin
  (~38%) to the readout. This is the summarization motif, shown causally, in a base model, without fine-tuning.
- **Without punctuation the summary route mostly disappears, and direct lookback takes over.** U retains only 9% under
  s0words, yet its no-knockout margin is as high as P's (+1.80 vs +1.67). In U the readout gets the mood almost
  entirely by reading the adjectives directly. Behaviorally nothing is lost; the *route* changes. Per the hypotheses: no
  strong relocation to later positions (U's residual +0.16 is small), so for this readout it's "the summary is not
  formed or not used, and retrieval compensates". Probes (milestone 3) will show whether the information is present at
  later positions but unused.
- **Capitals barely substitute** (U-cap 13% vs U 9%).
- **Misplaced punctuation still collects the mood** (M retains 34%; with only punctuation visible, M's +0.44 equals P's
  +0.43). Punctuation tokens act as collectors wherever they are, consistent with the France/Japan knockout.
- **K and the control leave the route intact** (K 49%, C_mid 51% vs P 51%). Stray capitals don't disrupt summarization;
  unusualness alone changes nothing.
- **Contrast with France/Japan:** country retained ≈10% in P (lookback-dominated); mood retains 51%. The type of property
  determines how much the model relies on summary points, which is a reportable result in itself.
- **Behavior contrast:** removing punctuation hurt the country readout (−0.45) but not mood (U ≥ P). For mood the model
  compensates fully through direct lookback, at least at these distances.

### 10.8 Milestone 3: first probe heatmaps, k=4 (2026-10-17)
Script: `scripts/m3_probes.py`; output: `outputs/m3/heatmap_{mood,country}_k4_{P,U}.png`, per-cell CSVs (table view)
and JSON summaries, cached activations in `outputs/m3/acts/`. 400 pairs (300 train / 100 test; test = held-out
templates, fillers and adjectives or cities), continuation readout, logistic probes (C=0.1, standardized features),
10,000 pair-level permutations of the test labels. Each run takes about 20 s after activations are cached; peak 6.7 GB.

**Method corrections made while building (decision log):**
- First version used ridge classifiers with a retrained-permutation null. The null's 95th percentile reached **1.00**
  in some cells: with a strong signal, permuted training labels stay slightly correlated with the truth, and the probe
  amplifies that (same failure as the single shuffle in 10.4). Replaced by a pair-level permutation of the *test* labels
  (no retraining), which is the correct null for held-out accuracy; its 95th percentile is 0.56–0.64 here.
- Ridge trailed logistic regression by up to 0.32 on some cells, so probes are logistic again (the brief's original
  choice). Probe strength: C ∈ {0.01, 0.1, 1} on 24 sample cells per property differ mostly by ≤ 0.03 with no consistent
  winner; C=0.1 kept.

**Findings (P):**
- *Sanity:* the L0 column is exactly 0.50 except at property words (1.0), as it must be: pair members have identical
  tokens elsewhere. Mood adjectives at L0 generalize to *held-out* adjectives (1.0), so valence is in the embeddings.
- *Punctuation spike (mood):* s0.period reaches 0.99 at L2 while the word before it is at 0.67; s0.comma2 0.84–1.00 vs
  pre_comma2 0.61–0.85; filler periods 0.73–0.98 vs the words before them 0.59–0.95. Country shows the same ordering
  more weakly through L9, then a global jump at L10–12 (most positions 0.95–1.0).
- *Saturation (risk 6.3, confirmed):* 92% (mood) and 90% (country) of cells beat the null; nearly every position after
  the property is decodable by mid-depth, even 4 sentences later.
- **Confound for comparing different rows:** punctuation and function-word positions are always the same token, whereas
  "the word before the comma" varies, so the probe's job is easier at fixed-identity positions. Consistent with that,
  the fixed word "and" after a comma scores as high as the comma. Claims that punctuation is *special* must therefore rest
  on causal methods (knockout, patching), or on matched fixed-identity comparisons. **Comparisons of the same row across
  conditions (same word, P vs U) are unaffected**, and that is the hypothesis test.

**Preview, U vs P at shared rows (mean accuracy over L1–4 / L5–8 / L9–12):**
- *Mood:* within ≈0.05 at almost every row (e.g. f1.pre_period 0.67/0.78/0.87 → 0.67/0.79/0.90; ro.final
  0.80/0.99/0.99 → 0.83/0.98/1.00); slightly lower early in the last filler (flast.first 0.84 → 0.77 at L1–4).
- *Country:* similar, with U lower late in the last filler and readout (flast.first 0.92 → 0.82; ro.q0 0.88 → 0.80 at L9–12).
- **No relocation signature:** words before a missing boundary (pre_comma, pre_period) do not become more decodable
  in U.
- **Synthesis with the knockout (10.7):** the label is *present* at later positions in both P and U (probes), but the
  readout *uses* mainly the property sentence's own punctuation (P: s0's commas and period carry ~38% of the margin; U
  retains only 9% via later positions). So it's "present but not read from there": decodability is diffuse, routing
  is selective. Without punctuation the model doesn't switch to reading the decodable later positions; it reads the
  adjectives directly. Probes alone would have suggested diffusion. This is the central methodological point for the paper.

### 10.9 Milestone 5: first patching heatmaps, mood k=4, P and U (2026-10-17)
Script: `scripts/m5_patching.py`; code: `punctsum/patching.py`; output: `outputs/m5/patching_mood_k4_{P,U}.{png,csv,json}`.
50 pairs, clean/corrupted alternating by pair, continuation readout, layers 1–12, recovery = ratio of mean gains with
a 95% bootstrap CI over pairs. Runs on the Apple GPU (MPS; logits match CPU to ~3e-5, about 4× faster). About 2 min per
configuration at k=4.
*Validation:* patching the final position at L12 reproduces the clean logit difference exactly; a same-token position
at L0 changes nothing; patching all positions at L6 reproduces the clean result; batched and single patches agree to
5 decimals; a one-element span equals the single-position patch.

**Single positions (P):** the three adjectives recover 0.40 / 0.23 / 0.36 at L1 (sum 0.99), fading by L8. **s0.period
recovers 0.08 (L5), 0.10 (L6), 0.14 (L7)**, then hands off to the readout (ro.final 0.13 at L7 → 0.57 at L9 → 1.0 at L12;
ro.q4 "experience" 0.17 at L8–9). s0's commas carry ≤ 0.04; every filler position ≈ 0.
**Single positions (U):** no position replaces the period (s0.pre_period 0.00 at every layer; no filler position above
0.01). Instead the adjectives keep their information longer (prop3: 0.20 vs 0.14 at L6, 0.13 vs 0.08 at L7; prop1 0.32
vs 0.27 at L6). The readout's profile is the same as in P.

**Spans (diffusion test), best layer:**

| span | P | U |
|---|---|---|
| s0 punctuation (2 commas + period) | 0.20 (L7) | — |
| all s0 tokens after the last adjective | 0.15 (L7) | 0.01 |
| all filler punctuation | 0.02 | — |
| all filler tokens | 0.06 (L7) | 0.05 (L8) |
| every passage token after the last adjective | 0.21 (L7) | 0.05 (L7–8) |
| readout tokens before the final one | 0.34 (L8) | 0.39 (L8) |

**Findings:**
- In P, the post-property route the readout uses runs almost entirely through s0's three punctuation marks (0.20 of
  0.21), peaking at L7. This matches the knockout (s0's punctuation ≈ 38% of the margin) and is the summarization motif
  measured by patching.
- In U the entire post-adjective passage carries ≤ 0.05 even when patched all at once. That rules out both
  **relocation** (no single substitute position) and **diffusion** (no thinly spread route) at this distance. The readout
  instead reads the adjectives directly, which hold their information longer. Behavior is unharmed (Section 10.7).
- **A fourth outcome, "bypass":** without punctuation the summary is simply not formed, and retrieval goes straight to the
  source with no behavioral cost. The brief's "degradation" hypothesis predicted a behavioral loss; here information is not
  lost, only the summary route. (Name proposed for the paper; refine once all k and conditions are in.)

### 10.10 Milestone 3: scaled probes, k ∈ {1, 4, 16} × 7 conditions × 2 properties (2026-10-17)
Script: `scripts/m3_probes.py --all` (42 configurations), summary `scripts/m3_summary.py`. Output: per-configuration
heatmaps, CSVs and JSONs in `outputs/m3/`; summaries `outputs/m3/summary_{mood,country}.png` and `summary.csv`
(Δ vs P at the same row, mean over L1–L12). About 25 min on CPU, peak 8.0 GB. Rows present in fewer than 80 train / 40
test passages are not probed (rare M rows such as a 6th misplaced mark; listed per run in the JSON).

- **Nulls behave everywhere:** 95th percentiles 0.61–0.67; 89–93% of cells beat them in every configuration (saturation).
- **Presence is barely affected by any condition:** |Δ vs P| ≤ 0.05 at almost every shared row, property and k.
- **Controls are flat:** C_mid and C_hi stay within ±0.05 (mostly ±0.02) of P, so unusualness doesn't change decodability. K
  is likewise close to P.
- **Consistent small losses in U and M** at the first word after a (missing) boundary and the first readout tokens:
  country flast.first −0.09 (k=4) and −0.07 (k=16) in both U and M; country ro.q0–q2 −0.04 to −0.05 in U, up to −0.10 in M;
  mood ro.q0/q1 −0.05 in U at k=16, down to −0.13 in M.
- **Faint relocation-like gain in mood U:** filler words before a missing boundary become slightly *more* decodable (+0.03
  to +0.06, e.g. f1.pre_period +0.06 at k=16). Patching spans (10.9) show these positions are not used by the readout.
- **U-cap** mostly tracks U, except where it keeps the sentence-initial capital (e.g. country flast.first: U −0.09, U-cap
  +0.00 at k=4). Capitals help mark where a new sentence starts.
- **Conclusion for the probe method:** presence is a weak discriminator here. The information is decodable almost everywhere
  in every condition; the differences between conditions are in which positions the readout *uses* (knockout,
  patching), which is where the hypotheses get decided.

### 10.11 Milestone 5: full patching, 42 configurations (2026-10-17)
Run on the H100 (`scripts/m5_patching.py --all --max-tokens 32768`, about 15 min; the Mac would have needed about 2.5 h).
Summary: `scripts/m5_summary.py` → `outputs/m5/summary.csv`, `summary_{mood,country}.png`; per-configuration heatmaps,
CSVs and JSONs in `outputs/m5/`. 50 pairs per configuration.
**Hardware consistency:** 15 configurations were computed on both the Mac (Apple GPU) and the H100 (CUDA)
(`outputs/m5_mac/`, `scripts/m5_crosscheck.py`, `outputs/m5/crosscheck.txt`): largest cell difference 3.8e-6 of the logit
gap; clean-vs-corrupted gaps identical to 4 decimals.

Peak recovery (share of the logit gap; best layer is L7 for mood, L10 for country; every cell's 95% CI excludes 0):

| mood | s0 punct span (k=1/4/16) | post-property passage | all fillers | property words at L7 |
|---|---|---|---|---|
| P | 0.15 / 0.20 / 0.16 | 0.16 / 0.21 / 0.18 | 0.04 / 0.06 / 0.06 | 0.24 / 0.28 / 0.33 |
| U | — | 0.03 / 0.05 / 0.07 | 0.03 / 0.05 / 0.06 | 0.28 / 0.42 / 0.52 |
| U-cap | — | 0.05 / 0.10 / 0.08 | 0.04 / 0.09 / 0.07 | 0.30 / 0.41 / 0.46 |
| M | 0.05 / 0.09 / 0.08 | 0.05 / 0.09 / 0.09 | 0.03 / 0.06 / 0.06 | 0.30 / 0.40 / 0.49 |
| K | 0.13 / 0.17 / 0.14 | 0.15 / 0.19 / 0.17 | 0.04 / 0.06 / 0.06 | 0.24 / 0.30 / 0.34 |
| C_mid | 0.15 / 0.19 / 0.16 | 0.16 / 0.21 / 0.19 | 0.04 / 0.06 / 0.07 | 0.24 / 0.29 / 0.35 |
| C_hi | 0.14 / 0.19 / 0.15 | 0.16 / 0.21 / 0.18 | 0.04 / 0.06 / 0.06 | 0.24 / 0.28 / 0.35 |

Country (all conditions, all k): city position ≈ 0.77–0.84 at L7; s0 punctuation ≤ 0.01; post-property passage 0.03–0.07;
fillers 0.01–0.05; clean-corrupted gap ≈ 8–10.7 logits.

**Findings (GPT-2 small, final for this model):**
- **Summary route (mood, P):** the property sentence's punctuation carries 15–20% of the gap at every distance (the
  period alone 11–14%), always peaking at L7 (of 12), then handing off to the readout tokens. Unaffected by unusualness
  (C_mid, C_hi ≈ P) and barely by dictation capitals (K: 13–17%).
- **Bypass (mood, U):** with no punctuation, the whole post-property passage carries only 3–7% and no single position more
  than 6%: no relocation and no diffusion. Reliance on the property words themselves grows instead (0.28 → 0.42 → 0.52 at
  L7 for k = 1/4/16, vs P's 0.24 → 0.33), and behavior is unharmed (10.7). The slight rise with k (0.03 → 0.07) is the only
  hint of a substitute route forming at long distance.
- **Capitals as a weak substitute:** U-cap recovers up to 10% (k=4), about half of P's route.
- **Misplaced punctuation is used about half as much** as correct punctuation (5–9% vs 15–20%), although the knockout
  showed it *can* carry the label when it's the only thing visible. Punctuation tokens collect information wherever they
  are, but the intact model routes mostly through those at true boundaries.
- **Country is lookback-dominated in every condition** (the city's own position carries ~80% at L7; summary routes ≤ 7%),
  confirming the knockout. The property type decides whether summary points are used at all.

## 11. Future research directions

### 11.1 Behavioral eval: do dictation-style stray capitals hurt large models? (deferred 2026-10-17)
The practical counterpart to research question 2. This project asks whether spurious capitals create
pseudo-boundaries *inside* small open models; this follow-up would ask whether they measurably lower task
accuracy in the large models people actually prompt. Not started: the user chose not to run it in parallel.
Design sketch:
- **Tasks:** 2–3 small public task sets with short natural-language prompts and automatically scorable answers
  (e.g. reading-comprehension QA, GSM8K-style word problems, instruction following).
- **Perturbation:** capitalize *n* random mid-sentence, lowercase, non-proper words per prompt, at a few rates
  (e.g. 1 per 50, 1 per 20, 1 per 10 words), seeded. Optional variant, if the dictation tool does this too:
  a stray capital preceded by a spurious period or comma.
- **Control:** a matched-unusualness perturbation that doesn't touch capitalization (word substitution, as in
  condition C), so "stray capitals hurt" can be told apart from "any noise hurts".
- **Models:** to be chosen; API models need keys and budget, so estimate calls and cost first.
- **Metric:** accuracy per rate, paired against the unperturbed prompt, with bootstrap confidence intervals.
- **Pilot evidence so far** (Section 10.2): in GPT-2 small, 3 stray capitals in ~20 words were as surprising as
  removing all punctuation but barely changed a recall answer.

### 11.2 Aggregated property (proposed 2026-10-17, pending decision)
Goal: a label that no single token reveals, so the readout cannot solve it by one lookback and has to rely on
aggregation, which is the regime the summarization motif is about. GPT-2 small's limited ability constrains the choice.
- **Preferred: mixed-valence mood.** The property sentence contains three mildly valenced words, e.g. two positive
  and one negative ("the hotel was cozy, the staff were rude, but the view was lovely"); the label is the majority
  valence. Any single word predicts the label only ~2/3 of the time; the full label needs combining. Readout
  (continuation): " overall the trip was" → " good" / " bad" (exact tokens to be screened).
- **Fallback if GPT-2 small can't do majority valence:** consistent-valence mood spread over 2–3 words (Tigges et
  al.'s setup). Each word is informative on its own, but the label is still spread over several tokens, and the
  result stays comparable to the paper that introduced the motif.
- **Acceptance test before any probing:** P accuracy well above chance, and a much larger share of P's margin
  surviving the s0words knockout than France/Japan's 5–10%. If neither version passes, report that GPT-2 small
  solves both properties by lookback, and move the aggregated property to Pythia.
- **As built (`punctsum/mood.py`, screen `scripts/m2b_mood_screen.py`, `outputs/m2b/screen_report.txt`):** 24
  three-clause templates ("At the hotel the room was ADJ all week, the staff was ADJ to us, and the breakfast was ADJ
  every morning"), each adjective followed by ≥2 words before the boundary. Answer tokens " great" / " terrible"
  (not " good"/" bad": GPT-2 small prefers " good" after almost anything, and " bad" is not even in its top 12 after
  "The food was awful. Overall, it was"). Adjectives must be single tokens with a valence (great−terrible shift
  relative to neutral adjectives) inside the band 1.2–2.6 for both signs: 22 positive (mean 1.88) and 14 negative
  (mean 1.74) kept, so the two sides have similar strength and a 2-vs-1 majority isn't overturned by one strong word.
  The 64 fillers closest to the median valence are used for mood passages (16 mildly valenced ones dropped).
  Templates, fillers and adjectives are each split train/test. Readout: " and overall the whole experience was" (best of 4).

**Pre-registered acceptance criteria (written 2026-10-17, before running `scripts/m2b_mood_accept.py`):** measured in
P with the continuation readout, pooled over k, 500 pairs (100 per k):
- **C1, GPT-2 can do it:** pair accuracy ≥ 0.90 **and** member-level AUC ≥ 0.80. AUC is threshold-free: the
  probability that a random "good" passage scores higher (great − terrible) than a random "bad" one. It is needed because
  a mixed pair differs in one adjective, so pair accuracy alone can pass without any majority aggregation.
- **C2, the summary route matters:** the s0words knockout retains ≥ 25% of P's no-knockout margin (France/Japan
  retained ≈10% with this readout type).
- **Decision:** if mixed passes C1 and C2, it becomes the main property. Otherwise, if consistent passes both, it does.
  If neither passes, report that and consider moving the aggregated property to Pythia. France/Japan stays as the
  lookback contrast either way.

## 12. References
- Chauhan, S., & Geiger, A. (2024). GPT-2 Small Fine-Tuned on Logical Reasoning Summarizes Information on
  Punctuation Tokens. NeurIPS 2024 Workshop. https://mlanthology.org/neuripsw/2024/chauhan2024neuripsw-gpt2
- Chauhan, S., Chaudhary, M., Choy, K., Nellessen, S., & Schoots, N. (2026). Punctuations and Predicates in
  Language Models. Findings of EACL 2026, 5622–5636. https://aclanthology.org/2026.findings-eacl.297/ ;
  arXiv:2508.14067
- Chen, G., et al. (2025). SepLLM: Accelerate Large Language Models by Compressing One Segment into One
  Separator. ICML 2025. arXiv:2412.12094 **[verify author list]**
- Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic Probing. TACL. arXiv:2006.00995
- Ferrando, J., & Voita, E. (2024). Information Flow Routes: Automatically Interpreting Language Models at
  Scale. arXiv:2403.00824
- Geva, M., Bastings, J., Filippova, K., & Globerson, A. (2023). Dissecting Recall of Factual Associations in
  Auto-Regressive Language Models. EMNLP 2023.
- Gong, Z., Sun, Y., Li, P., Zheng, Y., & Wang, Z. (2026). Punctuation-Steered Representation Fine-Tuning.
  ACL 2026 (short). https://aclanthology.org/2026.acl-short/
- Heimersheim, S., & Nanda, N. (2024). How to use and interpret activation patching. arXiv:2404.15255
- Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. EMNLP 2019.
- Liu, A., Srinivasan, D., & Penn, G. (2026). Some Tokens Behave like Magnets. EMNLP 2026. arXiv:2609.05743
- Razzhigaev, A., et al. (2025). LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory
  of Transformers. Findings of NAACL 2025. arXiv:2502.15007 **[verify author list]**
- Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024). Massive Activations in Large Language Models.
  arXiv:2402.17762
- Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities (2026). ACL 2026.
  arXiv:2604.10135 **[add authors]**
- Tigges, C., Hollinsworth, O. J., Geiger, A., & Nanda, N. (2023). Linear Representations of Sentiment in
  Large Language Models. arXiv:2310.15154
- Wang, L., et al. (2023). Label Words are Anchors: An Information Flow Perspective for Understanding
  In-Context Learning. EMNLP 2023. arXiv:2305.14160
- Zhang, F., & Nanda, N. (2023). Towards Best Practices of Activation Patching in Language Models: Metrics and
  Methods. arXiv:2309.16042
- (Unnamed authors) House of Cards: Massive Weights in LLMs. arXiv:2410.01866 **[add authors]**
