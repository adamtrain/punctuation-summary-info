# Where does summary information go when punctuation is removed?

Write-up: [Without punctuation, language models go back to the source](https://words.rat.foo/p/without-punctuation-language-models)

Language models use punctuation tokens as summary points: attention collects the gist of a clause onto the comma or
period that ends it, and later tokens read it from there (the "summarization motif", Tigges et al. 2023). This project
removes punctuation from the **input** and asks where that information goes: does it move to another position, spread
out, or get lost?

Short answer: mostly none of those. Without punctuation no single position takes over, and the readout reads the source
words directly instead ("bypass"); the information is still decodable further along, it just isn't read from there. The
larger Pythia models show a thin route spread across the passage. Over a few sentences removing punctuation costs little,
and how much depends on how the question is asked; over ~900–1,500 tokens it costs the larger models more (Llama-3.1-8B
loses more than half its mood margin). Tested on GPT-2 small, GPT-Neo-125M, Pythia 160M–6.9B and Llama-3.1-8B with
attention knockout, activation patching and linear probes.

## Where to look

- **[`BRIEF.md`](BRIEF.md)** is the lab notebook: design (§3–4), validity risks (§6), the pre-registered predictions for
  the scaling runs (§5.3), a dated decision log (§9) and every result with its numbers (§10). Start with §10.11
  (GPT-2, final), §10.14 (all models), §10.15 (long-distance) and §10.17 (post-hoc analyses for the write-up).
  It is a working notebook, kept as it was written: the project was run with Claude Code, which is why it opens with
  context for Claude and why some decision-log entries are marked [autonomous]; literature notes marked [verify] were
  not checked against the papers.
- **[`outputs/models/summary.md`](outputs/models/summary.md)** has one table with every headline number for all eight
  models, with 95% bootstrap CIs. **[`outputs/models/posthoc.md`](outputs/models/posthoc.md)** adds the long-distance
  margins by condition, per-model patching detail and the punctuation-only knockout.
- **[`INSTALL_LOG.md`](INSTALL_LOG.md)** records the environment, model downloads and the (now deleted) cloud GPU.

## Layout

```
punctsum/              library code
  materials.py, mood.py    templates, fillers, cities, adjectives
  dataset.py, roles.py     pair generation, train/test split, token roles
  conditions.py, longtext.py  P / U / U-cap / M / K / C renderings; PG19 long-distance excerpts
  model.py                 model loading (TransformerLens TransformerBridge), tokenization
  behavior.py              logit-difference scoring and attention knockout
  patching.py, probes.py   activation patching; logistic probes with permutation nulls
scripts/               one script per milestone step (m1_ … m6_), named after the BRIEF milestones
outputs/
  m1, m1b, m2, m2b, m3, m5   GPT-2 small, run on a Mac (canonical GPT-2 results)
  m5_mac, gpt2_h100           hardware cross-checks (Mac vs H100)
  m6                          model download manifest (pinned revisions) and setup logs
  models/<model>/             the frozen protocol per scaling model: behavior, knockout, acceptance test,
                              patching, probes, long-distance; summaries are the *_summary.json files
  models/{gpt2-items,pythia-ladder}/  shared item sets
```

Each step writes per-pair raw results (`*.jsonl`), a summary (`*_summary.json`) and usually a human-readable report
(`*_report.txt`). Heatmaps are PNGs next to the CSV/JSON they were drawn from.

## Conditions

| | |
|---|---|
| **P** | punctuated text |
| **U** | punctuation removed, sentence-initial words lowercased (proper nouns kept) |
| **U-cap** | punctuation removed, casing unchanged |
| **M** | the same number of commas and periods, placed at random non-boundary gaps |
| **K** | a few random mid-sentence words capitalized (dictation-style) |
| **C_lo / C_mid / C_hi** | control: random word substitution, tuned to 0.5× / 1× / 1.5× U's word-level surprisal |

## Reproducing

Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run python scripts/m6_models.py download gpt2
uv run python scripts/m6_scale.py run gpt2    # the frozen protocol on one model
```

The item sets (pairs, cities, adjectives) are committed, so runs reuse exactly the inputs reported here.
`scripts/m6_models.py download` fetches the weights at the exact revisions in `outputs/m6/models_manifest.json`
(Llama-3.1-8B is gated on Hugging Face). Everything is computed in fp32; the 7–8B models need an 80 GB GPU. The method
was frozen at git tag `method-freeze-v1`; later changes are logged in BRIEF.md §9 as deviations. The long-distance
condition needs the PG19 test split in `data/pg19_test/` (see INSTALL_LOG.md).

Not included, because of size: cached activations (`outputs/**/acts/`, ~60 GB; regenerate with the probe step) and
model weights.

## Notes for readers of the raw files

- `outputs/models/gpt-neo-125m/noop_*` are an invalid first knockout run (the hook didn't affect GPT-Neo's attention;
  BRIEF.md §9, 2026-10-18). They are kept for the record and not reported.
- Logs mention `scripts/m6_long_check.py`, an early tokenization gate replaced by `scripts/m6_token_check.py`.
- Paths under `/lambda/nfs/AT1` and `/home/ubuntu` in logs and queue scripts refer to the cloud GPU the scaling runs used.

## License

- **Code** (`punctsum/`, `scripts/`): MIT, see [`LICENSE`](LICENSE).
- **Data, results and documentation** (`outputs/`, `BRIEF.md`, `INSTALL_LOG.md`, this README): CC BY 4.0, see
  [`LICENSE-DATA`](LICENSE-DATA).

The long-distance filler (`outputs/models/long_excerpts.jsonl` and the `*_L900` / `*_L1500` pair files) is excerpted
from public-domain Project Gutenberg books in DeepMind's PG19 test split (Apache-2.0).
