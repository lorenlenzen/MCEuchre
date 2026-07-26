"""Export a trained PolicyValueNet checkpoint for the browser-based app.

Dumps every weight tensor as base64 float32 (row-major, PyTorch's native
(out_features, in_features) layout for Linear weights -- no transpose), in
the same {"shapes": [...], "data": [...]} shape the app already used for its
old (now-replaced) net, one entry per named submodule
(suit_encoder/context/make_trump/play_scorer/discard_scorer/pass_head/
value_head), each covering that submodule's params in a fixed order
(trunk.0.{weight,bias}, trunk.2.{weight,bias}, head.{weight,bias} for the
MLP submodules; weight, bias for the bare pass_head Linear). The JS side
(PolicyValueNetJS in application/euchre_vs_agents.html) must consume the
same order -- see that file's comments.

    python scripts/export_web_model.py --checkpoint checkpoints/rebel_sa.pt \
        --out application/model_weights.json

Then splice the result into the app with apply_web_model.py.
"""

import argparse
import base64
import json

import numpy as np
import torch

from rebel.networks import PolicyValueNet

# Order must match PolicyValueNetJS's constructor expectations.
_MLP_KEYS = ["trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
             "head.weight", "head.bias"]
_MLP_MODULES = ["suit_encoder", "context", "make_trump", "play_scorer",
                "discard_scorer", "value_head"]


def _b64_of(t: torch.Tensor):
    arr = t.detach().cpu().numpy().astype(np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii"), list(arr.shape)


def export(checkpoint_path: str) -> dict:
    net = PolicyValueNet()
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    net.eval()
    sd = net.state_dict()

    modules = {}
    for mod in _MLP_MODULES:
        shapes, data = [], []
        for key in _MLP_KEYS:
            b64, shape = _b64_of(sd[f"{mod}.{key}"])
            shapes.append(shape)
            data.append(b64)
        modules[mod] = {"shapes": shapes, "data": data}

    shapes, data = [], []
    for key in ["weight", "bias"]:
        b64, shape = _b64_of(sd[f"pass_head.{key}"])
        shapes.append(shape)
        data.append(b64)
    modules["pass_head"] = {"shapes": shapes, "data": data}

    return modules


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/rebel_sa.pt")
    ap.add_argument("--out", type=str, default="application/model_weights.json")
    args = ap.parse_args()

    modules = export(args.checkpoint)
    with open(args.out, "w") as f:
        json.dump(modules, f)
    total_bytes = sum(len(d) for m in modules.values() for d in m["data"])
    print(f"Exported {args.checkpoint} -> {args.out} "
          f"({total_bytes} base64 chars across {len(modules)} modules)")


if __name__ == "__main__":
    main()
