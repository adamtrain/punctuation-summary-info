"""Punctuation-removal summary-information experiment."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep model downloads inside the project so cleanup is one `rm -rf`. Must run before
# transformers / transformer_lens are imported.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
