// euchre_core.js — faithful JS port of euchre.py + encoder.py
// Mirrors the Python source structurally (same function names, same order of
// feature emission) so the trained weights' feature ordering lines up exactly.

const SUITS = "CDHS";
const RANKS = [9, 10, 11, 12, 13, 14];
const RANK_NAMES = { 9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A" };
const RANK_FROM_NAME = { "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14 };

function Card(rank, suit) { return { rank, suit }; }
function cardEq(a, b) { return a.rank === b.rank && a.suit === b.suit; }
function cardStr(c) { return `${RANK_NAMES[c.rank]}${c.suit}`; }
function cardFromStr(s) {
  const suit = s[s.length - 1];
  const rankPart = s.slice(0, -1);
  return Card(RANK_FROM_NAME[rankPart], suit);
}
function fullDeck() {
  const d = [];
  for (const s of SUITS) for (const r of RANKS) d.push(Card(r, s));
  return d;
}
function sameColorSuit(suit) {
  return { C: "S", S: "C", D: "H", H: "D" }[suit];
}
function isRightBower(c, trump) { return c.rank === 11 && c.suit === trump; }
function isLeftBower(c, trump) { return c.rank === 11 && c.suit === sameColorSuit(trump); }
function effectiveSuit(c, trump) { return isLeftBower(c, trump) ? trump : c.suit; }
function isTrump(c, trump) { return effectiveSuit(c, trump) === trump; }
function cardStrength(c, trump, ledSuit) {
  if (isRightBower(c, trump)) return 300;
  if (isLeftBower(c, trump)) return 200;
  if (isTrump(c, trump)) return 100 + c.rank;
  if (effectiveSuit(c, trump) === ledSuit) return c.rank;
  return -1;
}
function legalPlays(hand, trickCards, trump) {
  if (trickCards.length === 0) return hand.slice();
  const led = effectiveSuit(trickCards[0], trump);
  const follow = hand.filter((c) => effectiveSuit(c, trump) === led);
  return follow.length ? follow : hand.slice();
}
function trickWinner(plays, trump) {
  const led = effectiveSuit(plays[0][1], trump);
  let bestP = null, bestS = null;
  for (const [p, c] of plays) {
    const s = cardStrength(c, trump, led);
    if (bestS === null || s > bestS) { bestS = s; bestP = p; }
  }
  return bestP;
}
function teamOf(p) { return p % 2; }
function partnerOf(p) { return (p + 2) % 4; }

// ---------------------------------------------------------------------------
// Trump-relative suit roles (mirrors encoder.py exactly)
// ---------------------------------------------------------------------------

const ROLE_ORDER = ["trump", "next", "green1", "green2"];

function suitRoles(trump) {
  const nxt = sameColorSuit(trump);
  const greens = SUITS.split("").filter((s) => s !== trump && s !== nxt);
  return { [trump]: "trump", [nxt]: "next", [greens[0]]: "green1", [greens[1]]: "green2" };
}

const ROLE_RANKS = {
  trump: ["RB", "LB", "A", "K", "Q", "10", "9"],
  next: ["A", "K", "Q", "10", "9"],
  green1: ["A", "K", "Q", "J", "10", "9"],
  green2: ["A", "K", "Q", "J", "10", "9"],
};
const ROLE_BASE = {};
const SLOT_NAMES = [];
{
  let acc = 0;
  for (const r of ROLE_ORDER) {
    ROLE_BASE[r] = acc;
    for (const rk of ROLE_RANKS[r]) SLOT_NAMES.push(`${r}.${rk}`);
    acc += ROLE_RANKS[r].length;
  }
}
const N_SLOTS = SLOT_NAMES.length; // 24

function cardRoleRank(c, trump) {
  if (isRightBower(c, trump)) return ["trump", "RB"];
  if (isLeftBower(c, trump)) return ["trump", "LB"];
  const role = suitRoles(trump)[c.suit];
  return [role, RANK_NAMES[c.rank]];
}
function slotIndex(c, trump) {
  const [role, rank] = cardRoleRank(c, trump);
  return ROLE_BASE[role] + ROLE_RANKS[role].indexOf(rank);
}

// ---------------------------------------------------------------------------
// Seat-relative mapping
// ---------------------------------------------------------------------------

const REL_ORDER = ["me", "lopp", "partner", "ropp"];
function relOf(seat, other) { return REL_ORDER[((other - seat) % 4 + 4) % 4]; }

// ---------------------------------------------------------------------------
// Bidding-history reconstruction (mirrors reconstruct_bids exactly)
// ---------------------------------------------------------------------------

function reconstructBids(dealer, phase, currentPlayer, maker, alone, turnedDownSuit) {
  const order = [0, 1, 2, 3].map((i) => (dealer + 1 + i) % 4);
  const r1 = { 0: "none", 1: "none", 2: "none", 3: "none" };
  const r2 = { 0: "none", 1: "none", 2: "none", 3: "none" };

  function combine() {
    const out = {};
    for (let p = 0; p < 4; p++) out[p] = [r1[p], r2[p]];
    return out;
  }
  function fillPassesUntil(rounddict, stopper, stopToken) {
    for (const p of order) {
      if (p === stopper) { rounddict[p] = stopToken; return; }
      rounddict[p] = "pass";
    }
  }

  if (phase === "bid1") {
    for (const p of order) {
      if (p === currentPlayer) break;
      r1[p] = "pass";
    }
    return combine();
  }

  if (turnedDownSuit === null || turnedDownSuit === undefined) {
    const tok = alone ? "order_alone" : "order";
    fillPassesUntil(r1, maker, tok);
    return combine();
  }

  for (const p of order) r1[p] = "pass";
  if (phase === "bid2") {
    for (const p of order) {
      if (p === currentPlayer) break;
      r2[p] = "pass";
    }
    return combine();
  }

  const tok = alone ? "call_alone" : "call";
  fillPassesUntil(r2, maker, tok);
  return combine();
}

// ---------------------------------------------------------------------------
// Feature-vector builder (mirrors encoder.py's _FV + _build exactly)
// ---------------------------------------------------------------------------

class FV {
  constructor() { this.names = []; this.vals = []; }
  addBool(name, val) { this.names.push(name); this.vals.push(val ? 1.0 : 0.0); }
  add(name, val) { this.names.push(name); this.vals.push(Number(val)); }
  onehot(prefix, cats, active) {
    for (const c of cats) this.addBool(`${prefix}.${c}`, c === active);
  }
  multihot(prefix, cats, activeSet) {
    for (const c of cats) this.addBool(`${prefix}.${c}`, activeSet.has(c));
  }
}

function currentWinnerRel(info, trump) {
  if (info.current_trick.length === 0) return ["none", false, false];
  const led = effectiveSuit(info.current_trick[0][1], trump);
  let bestP = null, bestS = null;
  for (const [p, c] of info.current_trick) {
    const s = cardStrength(c, trump, led);
    if (bestS === null || s > bestS) { bestS = s; bestP = p; }
  }
  const rel = relOf(info.seat, bestP);
  return [rel, rel === "partner", rel === "me"];
}

function deriveVoids(info, trump) {
  const voids = { 0: new Set(), 1: new Set(), 2: new Set(), 3: new Set() };
  const roles = suitRoles(trump);
  const tricks = info.completed_tricks.slice();
  if (info.current_trick.length) tricks.push(info.current_trick);
  for (const plays of tricks) {
    if (!plays.length) continue;
    const ledRole = roles[effectiveSuit(plays[0][1], trump)];
    for (const [p, c] of plays) {
      if (roles[effectiveSuit(c, trump)] !== ledRole) voids[p].add(ledRole);
    }
  }
  const heldRoles = new Set(info.hand.map((c) => roles[effectiveSuit(c, trump)]));
  voids[info.seat] = new Set(ROLE_ORDER.filter((r) => !heldRoles.has(r)));
  return voids;
}

function upcardDisposition(info) {
  if (info.phase === "bid1") return "up";
  if (info.turned_down_suit !== null && info.turned_down_suit !== undefined) return "turned_down";
  return "ordered";
}

function buildFeatures(fv, info, trump) {
  const seat = info.seat;
  const roles = suitRoles(trump);
  const relseat = (p) => relOf(seat, p);

  fv.onehot("phase", ["bid1", "bid2", "discard", "play"], info.phase);
  fv.addBool("is_bidding", info.phase === "bid1" || info.phase === "bid2");
  fv.onehot("dealer_rel", REL_ORDER, relseat(info.dealer));
  const makerRel = info.maker !== null && info.maker !== undefined ? relseat(info.maker) : "none";
  fv.onehot("maker_rel", [...REL_ORDER, "none"], makerRel);
  fv.addBool("alone", info.alone);
  const sitRel = info.sitting_out !== null && info.sitting_out !== undefined ? relseat(info.sitting_out) : "none";
  fv.onehot("sitting_rel", [...REL_ORDER, "none"], sitRel);
  fv.addBool("i_am_maker", info.maker === seat);
  fv.addBool("partner_is_maker", info.maker !== null && info.maker !== undefined && partnerOf(seat) === info.maker);
  fv.addBool("loner_declare_ctx", info.alone && info.maker === seat);
  fv.addBool("loner_defense_ctx", info.alone && info.maker !== null && info.maker !== undefined && teamOf(info.maker) !== teamOf(seat));

  const bids = reconstructBids(info.dealer, info.phase, info.current_player,
    info.maker, info.alone, info.turned_down_suit);
  for (const rel of REL_ORDER) {
    const p = [0, 1, 2, 3].find((q) => relseat(q) === rel);
    const [r1, r2] = bids[p];
    fv.onehot(`bid_r1.${rel}`, ["none", "pass", "order", "order_alone"], r1);
    fv.onehot(`bid_r2.${rel}`, ["none", "pass", "call", "call_alone"], r2);
  }

  const handSlots = new Set(info.hand.map((c) => slotIndex(c, trump)));
  const playedBy = {};
  for (const [p, c] of info.played) playedBy[slotIndex(c, trump)] = p;
  const knownDead = new Set();
  for (const c of info.known_dead_extra) knownDead.add(slotIndex(c, trump));
  if (info.turned_down_suit !== null && info.turned_down_suit !== undefined) {
    knownDead.add(slotIndex(info.upcard, trump));
  }

  for (let i = 0; i < N_SLOTS; i++) fv.addBool(`hand.${SLOT_NAMES[i]}`, handSlots.has(i));

  for (let i = 0; i < N_SLOTS; i++) {
    const out_ = !handSlots.has(i) && !(i in playedBy) && !knownDead.has(i);
    fv.addBool(`outstanding.${SLOT_NAMES[i]}`, out_);
  }

  for (let i = 0; i < N_SLOTS; i++) {
    const who = (i in playedBy) ? relseat(playedBy[i]) : null;
    fv.multihot(`prov.${SLOT_NAMES[i]}`, REL_ORDER, who ? new Set([who]) : new Set());
  }

  const bowerCats = ["in_hand", "played_me", "played_lopp", "played_partner", "played_ropp", "outstanding"];
  function bowerState(slot) {
    if (handSlots.has(slot)) return "in_hand";
    if (slot in playedBy) return `played_${relseat(playedBy[slot])}`;
    return "outstanding";
  }
  fv.onehot("right_bower", bowerCats, bowerState(ROLE_BASE["trump"] + 0));
  fv.onehot("left_bower", bowerCats, bowerState(ROLE_BASE["trump"] + 1));

  const voids = deriveVoids(info, trump);
  for (const rel of REL_ORDER) {
    const p = [0, 1, 2, 3].find((q) => relseat(q) === rel);
    fv.multihot(`void.${rel}`, ROLE_ORDER, voids[p]);
  }

  const ledRole = info.current_trick.length ? roles[effectiveSuit(info.current_trick[0][1], trump)] : "none";
  fv.onehot("led_role", [...ROLE_ORDER, "none"], ledRole);
  const ncards = info.current_trick.length;
  fv.onehot("trick_ncards", [0, 1, 2, 3], Math.min(ncards, 3));
  for (let pos = 0; pos < 3; pos++) {
    if (pos < ncards) {
      const [p, c] = info.current_trick[pos];
      fv.addBool(`trick${pos}.filled`, true);
      fv.onehot(`trick${pos}.rel`, REL_ORDER, relseat(p));
      const si = slotIndex(c, trump);
      for (let i = 0; i < N_SLOTS; i++) fv.addBool(`trick${pos}.card.${SLOT_NAMES[i]}`, i === si);
    } else {
      fv.addBool(`trick${pos}.filled`, false);
      fv.onehot(`trick${pos}.rel`, REL_ORDER, null);
      for (let i = 0; i < N_SLOTS; i++) fv.addBool(`trick${pos}.card.${SLOT_NAMES[i]}`, false);
    }
  }
  const [winRel, partnerWinning, iWinning] = currentWinnerRel(info, trump);
  fv.onehot("winner_rel", [...REL_ORDER, "none"], winRel);
  fv.addBool("partner_winning", partnerWinning);
  fv.addBool("i_am_winning", iWinning);

  const myTeam = teamOf(seat);
  const myTr = info.tricks_won[myTeam], oppTr = info.tricks_won[1 - myTeam];
  fv.onehot("my_tricks", [0, 1, 2, 3, 4, 5], Math.min(myTr, 5));
  fv.onehot("opp_tricks", [0, 1, 2, 3, 4, 5], Math.min(oppTr, 5));
  fv.add("tricks_done", (info.tricks_won[0] + info.tricks_won[1]) / 5.0);
  const tgt = Math.max(1, info.target_score);
  fv.add("my_score", info.scores[myTeam] / tgt);
  fv.add("opp_score", info.scores[1 - myTeam] / tgt);

  fv.add("upcard_known", 1.0);
  const upSlot = slotIndex(info.upcard, trump);
  for (let i = 0; i < N_SLOTS; i++) fv.addBool(`upcard.${SLOT_NAMES[i]}`, i === upSlot);
  fv.onehot("upcard_disp", ["up", "ordered", "turned_down"], upcardDisposition(info));
  const tdRole = (info.turned_down_suit !== null && info.turned_down_suit !== undefined)
    ? suitRoles(trump)[info.turned_down_suit] : "none";
  fv.onehot("turned_down_role", [...ROLE_ORDER, "none"], tdRole);

  // -- derived: closeness to winning (appended LAST, same as encoder.py) --
  fv.add("score_diff", (info.scores[myTeam] - info.scores[1 - myTeam]) / tgt);

  // -- derived: protected high-card value, per role (mirrors encoder.py) --
  for (const role of ROLE_ORDER) {
    const ranksArr = ROLE_RANKS[role];
    const heldPositions = [];
    for (let i = 0; i < ranksArr.length; i++) {
      if (handSlots.has(ROLE_BASE[role] + i)) heldPositions.push(i);
    }
    let best = 0.0;
    for (const i of heldPositions) {
      let n = 0;
      for (let j = 0; j < i; j++) {
        const slot = ROLE_BASE[role] + j;
        if (!handSlots.has(slot) && !(slot in playedBy) && !knownDead.has(slot)) n++;
      }
      const k = heldPositions.filter((j) => j > i).length;
      const v = n > 0 ? 1.0 - (n / (n + 1)) * Math.pow(0.5, k) : 1.0;
      best = Math.max(best, v);
    }
    fv.add(`protected_value.${role}`, best);
  }
}

function encode(info, candidateTrump) {
  const fv = new FV();
  buildFeatures(fv, info, candidateTrump);
  return Float32Array.from(fv.vals);
}
function describeJS(info, candidateTrump) {
  const fv = new FV();
  buildFeatures(fv, info, candidateTrump);
  return fv.names.map((n, i) => [n, fv.vals[i]]);
}

// ---------------------------------------------------------------------------
// observation_tensor port (mirrors euchre/infoset.py's observation_tensor
// line-for-line -- see that file for the authoritative reference and the
// suit-agnostic/trump-relative design rationale). Layout: [global block (32)
// | 4 per-suit blocks, absolute slot order, role-relative contents (25 each)
// | 24 per-card feature blocks, card-id order (11 each)] = 396 total.
// ---------------------------------------------------------------------------

const RANK_INDEX = { 9: 0, 10: 1, 11: 2, 12: 3, 13: 4, 14: 5 };
const PHASE_INDEX = { bid1: 0, bid2: 1, discard: 2, play: 3 };  // 4=terminal, unused here
const N_PHASES = 5;
const GLOBAL_DIM = 32;
const SUIT_BLOCK_DIM = 25;
const CARD_FEAT_DIM = 11;
const NUM_SUITS_ = 4;
const NUM_CARDS_ = 24;
const SUIT_OFF_ = GLOBAL_DIM;
const CARD_OFF_ = GLOBAL_DIM + NUM_SUITS_ * SUIT_BLOCK_DIM;
const OBS_SIZE_ = GLOBAL_DIM + NUM_SUITS_ * SUIT_BLOCK_DIM + NUM_CARDS_ * CARD_FEAT_DIM;  // 396
// as-if-trump holdings, in strength order: right bower, left bower, A,K,Q,10,9
// (right/left bower handled separately below; this covers the A/K/Q/10/9 tail)
const TRUMP_HOLDING_RANKS = [14, 13, 12, 10, 9];

function relIdx2(me, other) { return ((other - me) % 4 + 4) % 4; }

function referenceSuit(info) {
  if (info.trump_settled) return info.trump_settled;
  if (info.upcard) return info.upcard.suit;
  return null;
}
function roleOf2(suit, ref) {
  if (ref === null) return 2;
  if (suit === ref) return 0;
  if (suit === sameColorSuit(ref)) return 1;
  return 2;
}
function cardIdOf(c) { return SUITS.indexOf(c.suit) * 6 + RANK_INDEX[c.rank]; }
function cardFromIdJS(cid) { return Card(RANKS[cid % 6], SUITS[Math.floor(cid / 6)]); }

function encodeObservation(info) {
  const v = new Float32Array(OBS_SIZE_);
  const me = info.seat;
  const hand = info.hand;
  const handHas = (rank, suit) => hand.some((c) => c.rank === rank && c.suit === suit);
  const trump = info.trump_settled || null;
  const ref = referenceSuit(info);
  const showUp = !!(info.upcard && !trump);
  const myTeam = teamOf(me);

  // ---- global block (32) ----
  let o = 0;
  v[o + (PHASE_INDEX[info.phase] ?? 4)] = 1.0; o += N_PHASES;
  v[o + relIdx2(me, info.dealer)] = 1.0; o += 4;
  if (info.maker === null || info.maker === undefined) v[o] = 1.0;
  else v[o + 1 + relIdx2(me, info.maker)] = 1.0;
  o += 5;
  v[o] = info.alone ? 1.0 : 0.0; o += 1;
  v[o] = info.tricks_won[myTeam] / 5.0;
  v[o + 1] = info.tricks_won[1 - myTeam] / 5.0;
  o += 2;
  v[o] = info.scores[myTeam] / 10.0;
  v[o + 1] = info.scores[1 - myTeam] / 10.0;
  o += 2;
  v[o] = (info.phase === "play" && info.current_trick.length === 0) ? 1.0 : 0.0; o += 1;
  v[o] = (info.bids_seen || 0) / 7.0; o += 1;
  v[o] = showUp ? 1.0 : 0.0; o += 1;
  if (showUp) v[o + RANK_INDEX[info.upcard.rank]] = 1.0;
  o += 6;
  if (info.current_trick.length) {
    const ledRole = roleOf2(effectiveSuit(info.current_trick[0][1], trump), ref);
    v[o + ledRole + 1] = 1.0;
  } else {
    v[o] = 1.0;
  }
  o += 4;

  // ---- per-suit blocks (4 x 25, absolute slot order) ----
  const playedCards = [];  // [relSeat, card]
  for (const plays of info.completed_tricks) {
    for (const [seat, c] of plays) playedCards.push([relIdx2(me, seat), c]);
  }
  for (const [seat, c] of info.current_trick) playedCards.push([relIdx2(me, seat), c]);

  for (let si = 0; si < SUITS.length; si++) {
    const s = SUITS[si];
    o = SUIT_OFF_ + si * SUIT_BLOCK_DIM;
    v[o + roleOf2(s, ref)] = 1.0; o += 3;
    v[o] = (info.turned_down_suit === s) ? 1.0 : 0.0; o += 1;
    v[o] = (trump === s) ? 1.0 : 0.0; o += 1;
    v[o] = handHas(11, s) ? 1.0 : 0.0;                       // right bower of s
    v[o + 1] = handHas(11, sameColorSuit(s)) ? 1.0 : 0.0;    // left bower of s
    for (let i = 0; i < TRUMP_HOLDING_RANKS.length; i++) {
      v[o + 2 + i] = handHas(TRUMP_HOLDING_RANKS[i], s) ? 1.0 : 0.0;
    }
    o += 7;
    let asIfTrumpCount = 0;
    for (const c of hand) if (isTrump(c, s)) asIfTrumpCount++;
    v[o] = asIfTrumpCount / 5.0; o += 1;
    const effHere = hand.filter((c) => effectiveSuit(c, trump) === s);
    for (const c of effHere) v[o + RANK_INDEX[c.rank]] = 1.0;
    o += 6;
    v[o] = effHere.length === 0 ? 1.0 : 0.0; o += 1;
    for (const [relSeat, c] of playedCards) {
      if (effectiveSuit(c, trump) === s) v[o + relSeat] += 1.0 / 5.0;
    }
    o += 4;
    let seen = effHere.length;
    for (const [, c] of playedCards) if (effectiveSuit(c, trump) === s) seen++;
    if (showUp && effectiveSuit(info.upcard, trump) === s) seen++;
    v[o] = seen / 7.0; o += 1;
  }

  // ---- per-card blocks (24 x 11, card-id order) ----
  for (let cid = 0; cid < NUM_CARDS_; cid++) {
    const c = cardFromIdJS(cid);
    o = CARD_OFF_ + cid * CARD_FEAT_DIM;
    v[o + RANK_INDEX[c.rank]] = 1.0; o += 6;
    v[o] = hand.some((h) => cardEq(h, c)) ? 1.0 : 0.0; o += 1;
    v[o] = (trump !== null && isRightBower(c, trump)) ? 1.0 : 0.0; o += 1;
    v[o] = (trump !== null && isLeftBower(c, trump)) ? 1.0 : 0.0; o += 1;
    v[o] = (trump !== null && isTrump(c, trump)) ? 1.0 : 0.0; o += 1;
    v[o] = playedCards.some(([, p]) => cardEq(p, c)) ? 1.0 : 0.0; o += 1;
  }

  return v;
}

const EuchreCore = {
  SUITS, RANKS, RANK_NAMES, Card, cardEq, cardStr, cardFromStr, fullDeck,
  sameColorSuit, isRightBower, isLeftBower, effectiveSuit, isTrump,
  cardStrength, legalPlays, trickWinner, teamOf, partnerOf,
  slotIndex, N_SLOTS, SLOT_NAMES, ROLE_ORDER, suitRoles,
  encode, describeJS, encodeObservation, cardIdOf, cardFromIdJS,
  GLOBAL_DIM, SUIT_BLOCK_DIM, CARD_FEAT_DIM, OBS_SIZE_,
};

if (typeof window !== "undefined") Object.assign(window, EuchreCore);
if (typeof module !== "undefined") module.exports = EuchreCore;
