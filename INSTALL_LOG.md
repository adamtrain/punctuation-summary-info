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

### 2026-10-17: git repository and GitHub remote (local machine)
- **What:** `git init` in the project folder; private GitHub repository
  https://github.com/adamtrain/punctuation-summary-info (remote `origin`). `.gitignore` excludes `.venv/`, `.hf_cache/`,
  `outputs/**/acts/` (activation caches, 18 GB so far) and Python caches. Rule: files over 10 MB are not committed
  without checking with the user first.
- **Cleanup:** `rm -rf .git .gitignore`; delete the GitHub repository from its settings page (or `gh repo delete
  adamtrain/punctuation-summary-info`).

## Remote: Lambda H100 instance (`ssh lambda`)
Persistent storage `/lambda/nfs/AT1` survives instance termination; the home directory and local disk do not. Cleanup
of the local-disk items is automatic when the instance is terminated.

### 2026-10-17: uv (remote)
- **What:** uv 0.12.21 (same version as the Mac) from Astral's official installer
  (`curl -LsSf https://astral.sh/uv/0.12.21/install.sh | sh`) into `~/.local/bin/uv` and `~/.local/bin/uvx`. The installer
  appended `. "$HOME/.local/bin/env"` to `~/.bashrc`, `~/.profile` and `~/.zshrc`.
- **Cleanup:** `rm ~/.local/bin/uv ~/.local/bin/uvx ~/.local/bin/env ~/.local/bin/env.fish`; delete that line from the
  three profile files.

### 2026-10-17: Python environment (remote, local disk)
- **What:** uv-managed CPython 3.13 (`~/.local/share/uv/python`, 108 MB); project environment at
  `~/venvs/punctuation-summary-info` (5.9 GB), created with `UV_PROJECT_ENVIRONMENT=$HOME/venvs/punctuation-summary-info uv
  sync --frozen --python 3.13` from the same `uv.lock` as the Mac (torch 2.14.1+cu130, transformer-lens 4.2.0); uv download
  cache `~/.cache/uv`.
- **Cleanup:** `rm -rf ~/venvs/punctuation-summary-info ~/.local/share/uv ~/.cache/uv` (or terminate the instance).

### 2026-10-17: project copy and model cache (remote, persistent storage)
- **What:** project copied with rsync to `/lambda/nfs/AT1/punctuation-summary-info` (excluding `.venv`, `.hf_cache`,
  activation caches). Hugging Face cache at `/lambda/nfs/AT1/punctuation-summary-info/.hf_cache` (set automatically by
  `punctsum/__init__.py`). Weights live in a shared content-addressed store (`.hf_cache/hub/blobs/`) that each model's
  folder links into. Models downloaded (safetensors only, pinned revisions in `outputs/m6/models_manifest.json`,
  95 GB in total):
  - `gpt2` (≈ 0.5 GB)
  - 2026-10-17, via `scripts/m6_models.py download`: EleutherAI/pythia-160m (0.37 GB), -410m (0.91), -1.4b (2.93), -2.8b
    (5.68), -6.9b (13.85), -12b (23.84); meta-llama/Llama-3.1-8B (16.06 GB, gated); google/gemma-2-9b (36.97 GB, gated,
    fp32 weights).
  - 2026-10-17, at the user's request (pre-download in case it's used): google/gemma-2-27b (108.91 GB, 24 shards, gated,
    fp32 weights, revision 938270f527). Cache total now 196 GB.
  - 2026-10-17: google/gemma-2-27b **deleted** (user decision; freed 108.9 GB via `huggingface_hub.scan_cache_dir()
    .delete_revisions(...)`). Cache total now 95 GB. Gemma-2-9B (37 GB) remains but is unused (Llama-3.1-8B was chosen).
- **Login:** the user ran `huggingface-cli login` on the instance themselves; the token stays at its default location
  (`~/.cache/huggingface/token`, local disk), and the code only points `HF_TOKEN_PATH` at it. To remove it:
  `huggingface-cli logout` (or terminate the instance).
- **Cleanup:** `rm -rf /lambda/nfs/AT1/punctuation-summary-info` (persistent storage is billed separately; check Lambda's
  pricing).

## Not installed (deliberately)
- No Homebrew packages, no global pip installs, no Jupyter (so far).
- No Hugging Face login or token stored yet. Gemma (milestone 6) will need `HF_TOKEN`; record it here
  when that happens, and where the token is stored.

## Full cleanup (everything this experiment added)
```bash
rm -rf .venv .hf_cache
uv cache prune
```
