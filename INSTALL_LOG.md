# Install log

Everything installed or downloaded for this experiment, and how to remove it. Append new entries;
don't edit old ones.

## Already present before this experiment (not installed by us; do not remove)
- `uv` 0.12.21 (Homebrew)
- uv-managed CPython 3.13.15 at `~/.local/share/uv/python/cpython-3.13-macos-aarch64-none`
- System Python 3.9.6 (`/usr/bin/python3`)

## Entries

### 2026-10-17: project environment
- **What:** project-local virtualenv `.venv/` (Python 3.13.15), created with `uv sync` from
  `pyproject.toml`. Direct dependencies: `transformer-lens`, `torch`, `scikit-learn`, `matplotlib`,
  `numpy`. Resolved versions are pinned in `uv.lock`. A full `pip freeze` (100 packages) is in
  `outputs/pip-freeze-2026-10-17.txt`. Key versions: torch 2.14.1, transformer-lens 4.2.0,
  transformers 5.20.1, scikit-learn 1.9.1, numpy 2.5.4, matplotlib 3.11.2.
- **Size:** `.venv/` ≈ 1.1 GB.
- **Side effect outside the project:** uv's shared download cache `~/.cache/uv` grew from ≈ 1.5 GB to
  ≈ 2.5 GB (wheels for torch etc.). This cache is shared with other uv projects on this machine.
- **Cleanup:**
  - `rm -rf .venv` (in the project folder)
  - Optional, affects all uv projects: `uv cache prune` removes unused cache entries; `uv cache clean`
    empties the whole cache.

### 2026-10-17: Hugging Face model cache (project-local)
- **What:** `HF_HOME` points to `./.hf_cache` (set in `punctsum/__init__.py` and `.env`), so model weights
  and tokenizers download into the project, not `~/.cache/huggingface`.
- **Models downloaded:** see the list below; update it as models are added.
  - `gpt2` (GPT-2 small, ≈ 0.5 GB)
- **Cleanup:** `rm -rf .hf_cache`

## Not installed (deliberately)
- No Homebrew packages, no global pip installs, no Jupyter (so far).
- No Hugging Face login or token stored yet. Gemma (milestone 6) will need `HF_TOKEN`; record it here
  when that happens, and where the token is stored.

## Full cleanup (everything this experiment added)
```bash
rm -rf .venv .hf_cache
uv cache prune
```
