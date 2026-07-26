"""Export a trained PolicyValueNet checkpoint for the browser-based app.

Dumps every weight tensor as base64 float32 (row-major, PyTorch's native
(out_features, in_features) layout for Linear weights -- no transpose), in
the same {"shapes": [...], "data": [...]} shape the app already used for its
old (now-replaced) net, one entry per named submodule
(suit_encoder/context/make_trump/play_scorer/discard_scorer/pass_head/
value_head), each covering that submodule's params in a fixed order
(trunk.0.{weight,bias}, trunk.2.{weight,bias}, head.{weight,bias} for the
MLP submodules; weight, bias for the bare pass_head Linear). The JS side
(PolicyValueNetJS in application/Euchre vs Agents/js/nn_core.js) must
consume the same order -- see that file's comments.

Writes a .js file (`const WEIGHTS_V2 = {...};`), not .json -- the app's
euchre_vs_agents.html loads it via a plain
<script src="js/model_weights.js"> tag (js/euchre_game.js's initNets()
reads the resulting global), which works with the page opened directly as
a file:// URL. A fetch()'d .json would not: browsers block that under
file://, and requiring a local server just to view the app would defeat
the point of it being a self-contained folder.

To update the app to a newer checkpoint: run this (defaults already point
at the app's own file), then just reload the page -- no other step, no
splicing into the HTML.

    python scripts/export_web_model.py
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
    ap.add_argument("--out", type=str,
                    default="application/Euchre vs Agents/js/model_weights.js")
    args = ap.parse_args()

    modules = export(args.checkpoint)
    with open(args.out, "w") as f:
        f.write("const WEIGHTS_V2 = ")
        json.dump(modules, f, separators=(",", ":"))
        f.write(";\n")
    total_bytes = sum(len(d) for m in modules.values() for d in m["data"])
    print(f"Exported {args.checkpoint} -> {args.out} "
          f"({total_bytes} base64 chars across {len(modules)} modules)")


if __name__ == "__main__":
    main()
