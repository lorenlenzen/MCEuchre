// nn_core.js — forward-pass-only JS port of nn.py's ActorCriticNet.
// Loads trained weights (base64 float32, row-major, matching numpy .tobytes()).

function b64ToFloat32Array(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

function relu(v) {
  const y = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) y[i] = Math.max(0, v[i]);
  return y;
}

// dense layer using PyTorch's native nn.Linear.weight layout: flat row-major
// (out_features, in_features), i.e. W[j*nin + i] is the weight from input i
// to output j. No transpose needed between the exported checkpoint and this
// -- matches scripts/export_web_model.py's straight .numpy().tobytes().
function denseForwardOI(x, W, b, nin, nout) {
  const y = new Float32Array(nout);
  for (let j = 0; j < nout; j++) {
    let acc = b[j];
    const base = j * nin;
    for (let i = 0; i < nin; i++) acc += x[i] * W[base + i];
    y[j] = acc;
  }
  return y;
}

// Forward-pass-only port of rebel/networks.py's MLP: Linear->ReLU->Linear->
// ReLU->Linear(head), matching depth=2 (the depth every submodule in
// PolicyValueNet uses). params/shapes: [t0.W,t0.b,t2.W,t2.b,head.W,head.b],
// shapes in PyTorch (out,in) order -- see export_web_model.py's _MLP_KEYS.
class MLPJS {
  constructor(params, shapes) { this.p = params; this.s = shapes; }
  forward(x) {
    const [t0w, t0b, t2w, t2b, hw, hb] = this.p;
    const [s0, , s2, , sh] = this.s;
    let a = relu(denseForwardOI(x, t0w, t0b, s0[1], s0[0]));
    a = relu(denseForwardOI(a, t2w, t2b, s2[1], s2[0]));
    return denseForwardOI(a, hw, hb, sh[1], sh[0]);
  }
}

// Forward-pass-only port of rebel/networks.py's PolicyValueNet (the
// suit-agnostic relational architecture) -- see that file's forward() for
// the reference this mirrors line-for-line. Built from a WEIGHTS_V2-shaped
// object (produced by scripts/export_web_model.py): one {shapes,data} entry
// per named submodule (suit_encoder/context/make_trump/play_scorer/
// discard_scorer/pass_head/value_head).
class PolicyValueNetJS {
  constructor(w) {
    const build = (e) => new MLPJS(e.data.map(b64ToFloat32Array), e.shapes);
    this.suit_encoder = build(w.suit_encoder);
    this.context = build(w.context);
    this.make_trump = build(w.make_trump);
    this.play_scorer = build(w.play_scorer);
    this.discard_scorer = build(w.discard_scorer);
    this.value_head = build(w.value_head);
    this.pass_w = b64ToFloat32Array(w.pass_head.data[0]);
    this.pass_b = b64ToFloat32Array(w.pass_head.data[1]);
    this.pass_shape = w.pass_head.shapes[0];  // [1, context_dim]
  }

  forward(obs) {
    const glob = obs.subarray(0, GLOBAL_DIM);
    const suitBlocks = [];
    for (let s = 0; s < NUM_SUITS_; s++) {
      suitBlocks.push(obs.subarray(SUIT_OFF_ + s * SUIT_BLOCK_DIM,
                                    SUIT_OFF_ + (s + 1) * SUIT_BLOCK_DIM));
    }
    const cardFeats = [];
    for (let c = 0; c < NUM_CARDS_; c++) {
      cardFeats.push(obs.subarray(CARD_OFF_ + c * CARD_FEAT_DIM,
                                   CARD_OFF_ + (c + 1) * CARD_FEAT_DIM));
    }
    // role one-hot's first dim of each suit block is the reference indicator.
    const refInd = suitBlocks.map((b) => b[0]);

    const suitE = suitBlocks.map((b) => this.suit_encoder.forward(b));
    const E = suitE[0].length;
    const pooled = new Float32Array(E);
    for (let e = 0; e < E; e++) {
      let sum = 0;
      for (let s = 0; s < NUM_SUITS_; s++) sum += suitE[s][e];
      pooled[e] = sum / NUM_SUITS_;
    }
    const ctxIn = new Float32Array(E + GLOBAL_DIM);
    ctxIn.set(pooled, 0);
    ctxIn.set(glob, E);
    const ctx = this.context.forward(ctxIn);
    const C = ctx.length;

    let orderupNa = 0, orderupAl = 0;
    const callNa = new Float32Array(NUM_SUITS_);
    const callAl = new Float32Array(NUM_SUITS_);
    for (let s = 0; s < NUM_SUITS_; s++) {
      const mtIn = new Float32Array(E + C);
      mtIn.set(suitE[s], 0);
      mtIn.set(ctx, E);
      const out = this.make_trump.forward(mtIn);
      callNa[s] = out[0];
      callAl[s] = out[1];
      orderupNa += refInd[s] * out[0];
      orderupAl += refInd[s] * out[1];
    }

    const play = new Float32Array(NUM_CARDS_);
    const discard = new Float32Array(NUM_CARDS_);
    for (let c = 0; c < NUM_CARDS_; c++) {
      const suitIdx = Math.floor(c / 6);  // card ids are suit*6+rank order
      const cardIn = new Float32Array(E + CARD_FEAT_DIM + C);
      cardIn.set(suitE[suitIdx], 0);
      cardIn.set(cardFeats[c], E);
      cardIn.set(ctx, E + CARD_FEAT_DIM);
      play[c] = this.play_scorer.forward(cardIn)[0];
      discard[c] = this.discard_scorer.forward(cardIn)[0];
    }

    const passLogit = denseForwardOI(ctx, this.pass_w, this.pass_b,
                                     this.pass_shape[1], this.pass_shape[0])[0];
    const value = this.value_head.forward(ctx)[0];

    // Assembled in euchre/actions.py's flat index order: play[0:24],
    // discard[24:48], call[48:52], call_alone[52:56], orderup[56],
    // orderup_alone[57], pass[58].
    const logits = new Float32Array(59);
    logits.set(play, 0);
    logits.set(discard, 24);
    logits.set(callNa, 48);
    logits.set(callAl, 52);
    logits[56] = orderupNa;
    logits[57] = orderupAl;
    logits[58] = passLogit;
    return { logits, value };
  }
}

// masked softmax over legal actions -> probabilities (illegal = 0)
function maskedSoftmax(logits, mask) {
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) if (mask[i] && logits[i] > mx) mx = logits[i];
  const ez = new Float64Array(logits.length);
  let Z = 0;
  for (let i = 0; i < logits.length; i++) {
    if (mask[i]) { ez[i] = Math.exp(logits[i] - mx); Z += ez[i]; }
  }
  const p = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) p[i] = mask[i] ? ez[i] / Z : 0;
  return p;
}
function argmaxMasked(vals, mask) {
  let best = -1, bestV = -Infinity;
  for (let i = 0; i < vals.length; i++) {
    if (mask[i] && vals[i] > bestV) { bestV = vals[i]; best = i; }
  }
  return best;
}
function sampleWeighted(probs, rngFn) {
  const r = (rngFn ? rngFn() : Math.random());
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r <= acc) return i;
  }
  return probs.length - 1;
}

const NNCore = { maskedSoftmax, argmaxMasked, sampleWeighted, PolicyValueNetJS };
if (typeof window !== "undefined") Object.assign(window, NNCore);
if (typeof module !== "undefined") module.exports = NNCore;
