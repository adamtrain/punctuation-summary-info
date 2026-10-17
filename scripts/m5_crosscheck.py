"""Hardware consistency check for patching: Mac (Apple GPU, outputs/m5_mac/) vs H100 (CUDA, outputs/m5/).

For every configuration computed on both machines, compare recovery cell by cell (same pairs, rows and layers).
Writes outputs/m5/crosscheck.txt.
"""

import json

import numpy as np

from punctsum import PROJECT_ROOT

MAC, GPU = PROJECT_ROOT / "outputs" / "m5_mac", PROJECT_ROOT / "outputs" / "m5"


def main():
    lines = ["Mac (MPS) vs H100 (CUDA) patching recovery, per configuration: max and mean |difference| over all cells\n"]
    worst = 0.0
    for mac_path in sorted(MAC.glob("patching_*.json")):
        gpu_path = GPU / mac_path.name
        if not gpu_path.exists():
            continue
        a, b = json.loads(mac_path.read_text()), json.loads(gpu_path.read_text())
        assert a["rows"] == b["rows"] and a["layers"] == b["layers"], mac_path.name
        d = np.abs(np.array(a["recovery"]) - np.array(b["recovery"]))
        worst = max(worst, d.max())
        lines.append(f"  {a['tag']:20} cells {d.size:4}  max {d.max():.2e}  mean {d.mean():.2e}  "
                     f"gap {a['mean_gap']:+.4f} vs {b['mean_gap']:+.4f}")
    lines.append(f"\nWorst cell difference across all configurations: {worst:.2e} (recovery is a share of the logit gap)")
    (GPU / "crosscheck.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
