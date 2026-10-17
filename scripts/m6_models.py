"""Milestone 6: check and download the scaling models (BRIEF.md section 5.2). Run on the cloud GPU machine.

  python scripts/m6_models.py check              # no download: access, pinned revision, weight format and size
  python scripts/m6_models.py download [names]   # download weights + configs + tokenizer into the project's HF cache

Only safetensors weights are downloaded (never pickled .bin files, which can execute code when loaded). The exact
revision (commit hash) of every model is recorded in outputs/m6/models_manifest.json, so results can be tied to the
precise weights used. Base (non-instruct) models only, to match GPT-2 and Pythia.
"""

import json
import sys

from punctsum import PROJECT_ROOT  # noqa: F401  (sets HF_HOME before huggingface_hub is imported)
from huggingface_hub import HfApi, snapshot_download

MODELS = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "pythia-12b": "EleutherAI/pythia-12b",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-27b": "google/gemma-2-27b",
}
OUT = PROJECT_ROOT / "outputs" / "m6"
KEEP = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"]


def check() -> dict:
    api = HfApi()
    out = {}
    for name, repo in MODELS.items():
        try:
            info = api.model_info(repo, files_metadata=True)
        except Exception as e:  # gated without access, or missing
            out[name] = {"repo": repo, "ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
            print(f"{name:13} {repo:28} NO ACCESS: {type(e).__name__}")
            continue
        files = {s.rfilename: s.size or 0 for s in info.siblings}
        st = sum(v for k, v in files.items() if k.endswith(".safetensors"))
        binsz = sum(v for k, v in files.items() if k.endswith(".bin"))
        out[name] = {"repo": repo, "ok": st > 0, "revision": info.sha, "gated": bool(info.gated),
                     "safetensors_gb": round(st / 1e9, 2), "bin_gb": round(binsz / 1e9, 2)}
        print(f"{name:13} {repo:28} rev {info.sha[:10]} gated={str(bool(info.gated)):5} "
              f"safetensors {st / 1e9:6.1f} GB  (.bin {binsz / 1e9:6.1f} GB)")
    return out


def download(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "models_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    status = check()
    for name in names:
        s = status[name]
        if not s["ok"]:
            print(f"skip {name}: {s.get('error', 'no safetensors weights')}", flush=True)
            continue
        path = snapshot_download(s["repo"], revision=s["revision"], allow_patterns=KEEP)
        manifest[name] = {**s, "local_path": path}
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"downloaded {name} @ {s['revision'][:10]} ({s['safetensors_gb']} GB)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    elif sys.argv[1] == "download":
        download(sys.argv[2:] or list(MODELS))
