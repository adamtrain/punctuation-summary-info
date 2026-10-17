"""Punctuation-removal summary-information experiment."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep model downloads inside the project so cleanup is one `rm -rf`. Must run before
# transformers / transformer_lens are imported.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
# ...but keep reading the login token from its default place (`huggingface-cli login` writes it there), which gated
# models (Llama, Gemma) need. The token is never copied into the project.
os.environ.setdefault("HF_TOKEN_PATH", str(Path.home() / ".cache" / "huggingface" / "token"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
