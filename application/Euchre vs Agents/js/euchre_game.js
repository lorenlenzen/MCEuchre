// euchre_game.js — game state machine + AI decision logic (ports of
// euchre.py's EuchreGame, heuristic.py's estimate_tricks/_choose_discard, and
// bidding.py's loner_value_estimate/net_bid_action), wired to the trained nets.

const PHASE_BID1 = "bid1", PHASE_BID2 = "bid2", PHASE_DISCARD = "discard",
  PHASE_PLAY = "play", PHASE_HAND_DONE = "hand_done", PHASE_GAME_DONE = "game_done";

const ALONE_VALUE_TH = 2.2;

class SimpleRNG {
  constructor(seed) { this.s = seed >>> 0 || 123456789; }
  next() { // xorshift32
    let x = this.s;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this.s = x >>> 0;
    return (this.s >>> 0) / 4294967296;
  }
  int(n) { return Math.floor(this.next() * n); }
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}

// ---------------------------------------------------------------------------
// EuchreGame — mirrors euchre.py's EuchreGame
// ---------------------------------------------------------------------------

class EuchreGame {
  constructor(targetScore, dealer, rng) {
    this.target_score = targetScore;
    this.rng = rng;
    this.scores = [0, 0];
    this.dealer = dealer;
    this._startHand();
  }

  _startHand(deal) {
    if (deal) {
      this.hands = { 0: deal.hands[0].slice(), 1: deal.hands[1].slice(), 2: deal.hands[2].slice(), 3: deal.hands[3].slice() };
      this.upcard = deal.upcard;
      if (deal.dealer !== undefined) this.dealer = deal.dealer;
    } else {
      const deck = this.rng.shuffle(fullDeck());
      this.hands = {};
      for (let p = 0; p < 4; p++) this.hands[p] = deck.slice(p * 5, (p + 1) * 5);
      this.upcard = deck[20];
    }
    this.trump = null;
    this.maker = null;
    this.alone = false;
    this.sitting_out = null;
    this.turned_down_suit = null;
    this._bid_player = (this.dealer + 1) % 4;
    this._bid_passes = 0;
    this.leader = null;
    this.current_trick = [];
    this.tricks_won = [0, 0];
    this.tricks_won_by_seat = [0, 0, 0, 0];
    this.trick_history = [];
    this.played_cards = [];
    this.dealer_discard = null;
    this.phase = PHASE_BID1;
    this.current_player = this._bid_player;
    this.last_hand_result = null;
    this._bidsSeenFrozen = 0;
  }

  // Mirrors euchre/game.py's EuchreState.bids_seen: live during bidding
  // (count of players who've already acted this round), then frozen at
  // whatever it was the instant the deciding OrderUp/Call was taken -- it is
  // NOT reset or recomputed during discard/play, matching the Python engine
  // exactly (see _stepBid1/_stepBid2's _bidsSeenFrozen writes below).
  get bidsSeen() {
    if (this.phase === PHASE_BID1 || this.phase === PHASE_BID2) {
      return ((this.current_player - this.dealer - 1) % 4 + 4) % 4;
    }
    return this._bidsSeenFrozen;
  }

  activePlayers() {
    const out = [];
    for (let p = 0; p < 4; p++) if (p !== this.sitting_out) out.push(p);
    return out;
  }
  _nextActive(pos) {
    let n = (pos + 1) % 4;
    while (n === this.sitting_out) n = (n + 1) % 4;
    return n;
  }
  _firstActiveFrom(pos) {
    let n = pos;
    while (n === this.sitting_out) n = (n + 1) % 4;
    return n;
  }

  legalActions() {
    if (this.phase === PHASE_BID1) return [["pass"], ["order_up", false], ["order_up", true]];
    if (this.phase === PHASE_BID2) {
      const actions = [];
      const isDealer = this.current_player === this.dealer;
      if (!isDealer) actions.push(["pass"]);
      for (const s of SUITS) {
        if (s === this.turned_down_suit) continue;
        actions.push(["call", s, false]);
        actions.push(["call", s, true]);
      }
      return actions;
    }
    if (this.phase === PHASE_DISCARD) return this.hands[this.dealer].map((c) => ["discard", c]);
    if (this.phase === PHASE_PLAY) {
      const hand = this.hands[this.current_player];
      const trickCards = this.current_trick.map(([, c]) => c);
      return legalPlays(hand, trickCards, this.trump).map((c) => ["play", c]);
    }
    return [];
  }

  step(action) {
    if (this.phase === PHASE_BID1) return this._stepBid1(action);
    if (this.phase === PHASE_BID2) return this._stepBid2(action);
    if (this.phase === PHASE_DISCARD) return this._stepDiscard(action);
    if (this.phase === PHASE_PLAY) return this._stepPlay(action);
    throw new Error("cannot step in phase " + this.phase);
  }

  _stepBid1(action) {
    if (action[0] === "pass") {
      this._bid_passes++;
      if (this._bid_passes === 4) {
        this.turned_down_suit = this.upcard.suit;
        this.phase = PHASE_BID2;
        this._bid_player = (this.dealer + 1) % 4;
        this.current_player = this._bid_player;
        this._bid_passes = 0;
      } else {
        this._bid_player = (this._bid_player + 1) % 4;
        this.current_player = this._bid_player;
      }
      return;
    }
    if (action[0] === "order_up") {
      const alone = action[1];
      this._bidsSeenFrozen = this.bidsSeen;
      this.trump = this.upcard.suit;
      this.maker = this.current_player;
      this.alone = alone;
      if (alone) this.sitting_out = partnerOf(this.maker);
      this._enterDiscardOrPlay(true);
      return;
    }
    throw new Error("illegal bid1 action");
  }

  _stepBid2(action) {
    if (action[0] === "pass") {
      if (this.current_player === this.dealer) throw new Error("dealer cannot pass (stick-the-dealer)");
      this._bid_player = (this._bid_player + 1) % 4;
      this.current_player = this._bid_player;
      return;
    }
    if (action[0] === "call") {
      const [, suit, alone] = action;
      if (suit === this.turned_down_suit) throw new Error("cannot name turned-down suit");
      this._bidsSeenFrozen = this.bidsSeen;
      this.trump = suit;
      this.maker = this.current_player;
      this.alone = alone;
      if (alone) this.sitting_out = partnerOf(this.maker);
      this._enterDiscardOrPlay(false);
      return;
    }
    throw new Error("illegal bid2 action");
  }

  _enterDiscardOrPlay(dealerPicksUp) {
    if (dealerPicksUp) {
      const dealerSitsOut = this.sitting_out === this.dealer;
      if (!dealerSitsOut) {
        this.hands[this.dealer].push(this.upcard);
        this.phase = PHASE_DISCARD;
        this.current_player = this.dealer;
        return;
      }
    }
    this._beginPlay();
  }

  _stepDiscard(action) {
    const card = action[1];
    const h = this.hands[this.dealer];
    const idx = h.findIndex((c) => cardEq(c, card));
    if (idx < 0) throw new Error("discard must be in dealer's hand");
    h.splice(idx, 1);
    this.dealer_discard = card;
    this._beginPlay();
  }

  _beginPlay() {
    this.phase = PHASE_PLAY;
    this.leader = this._firstActiveFrom((this.dealer + 1) % 4);
    this.current_player = this.leader;
    this.current_trick = [];
  }

  _stepPlay(action) {
    const card = action[1];
    const hand = this.hands[this.current_player];
    const legal = legalPlays(hand, this.current_trick.map(([, c]) => c), this.trump);
    if (!legal.some((c) => cardEq(c, card))) throw new Error("illegal card");
    const idx = hand.findIndex((c) => cardEq(c, card));
    hand.splice(idx, 1);
    this.current_trick.push([this.current_player, card]);
    this.played_cards.push([this.current_player, card]);
    if (this.current_trick.length === this.activePlayers().length) this._resolveTrick();
    else this.current_player = this._nextActive(this.current_player);
  }

  _resolveTrick() {
    const winner = trickWinner(this.current_trick, this.trump);
    this.tricks_won[teamOf(winner)]++;
    this.tricks_won_by_seat[winner]++;
    this.trick_history.push({ plays: this.current_trick.slice(), winner });
    this.current_trick = [];
    const total = this.tricks_won[0] + this.tricks_won[1];
    if (total === 5) this._scoreHand();
    else { this.leader = winner; this.current_player = winner; }
  }

  _scoreHand() {
    const makersTeam = teamOf(this.maker);
    const defendersTeam = 1 - makersTeam;
    const makerTricks = this.tricks_won[makersTeam];
    let points, scoringTeam;
    if (makerTricks >= 3) {
      points = makerTricks === 5 ? (this.alone ? 4 : 2) : 1;
      this.scores[makersTeam] += points;
      scoringTeam = makersTeam;
    } else {
      points = 2;
      this.scores[defendersTeam] += points;
      scoringTeam = defendersTeam;
    }
    this.last_hand_result = {
      maker: this.maker, makers_team: makersTeam, trump: this.trump, alone: this.alone,
      maker_tricks: makerTricks, points, scoring_team: scoringTeam, euchred: makerTricks < 3,
    };
    this.phase = (Math.max(...this.scores) >= this.target_score) ? PHASE_GAME_DONE : PHASE_HAND_DONE;
  }

  dealNextHand() {
    this.dealer = (this.dealer + 1) % 4;
    this._startHand();
  }

  get done() { return this.phase === PHASE_GAME_DONE; }
  get winnerTeam() { return this.done ? (this.scores[0] >= this.target_score ? 0 : 1) : null; }
}

// ---------------------------------------------------------------------------
// heuristic.py ports: estimate_tricks, _choose_discard, card_value
// ---------------------------------------------------------------------------

function cardValue(c, trump) {
  if (isRightBower(c, trump)) return 300;
  if (isLeftBower(c, trump)) return 200;
  if (isTrump(c, trump)) return 100 + c.rank;
  return c.rank;
}
function nontrumpSuits(trump) { return SUITS.split("").filter((s) => s !== trump); }

function estimateTricks(hand, trump) {
  const trumps = hand.filter((c) => isTrump(c, trump));
  const offs = hand.filter((c) => !isTrump(c, trump));
  const nTrump = trumps.length;
  let est = 0.0;
  for (const c of trumps) {
    if (isRightBower(c, trump)) est += 0.95;
    else if (isLeftBower(c, trump)) est += 0.85;
    else if (c.rank === 14) est += 0.72;
    else if (c.rank === 13) est += 0.50;
    else if (c.rank === 12) est += 0.30;
    else if (c.rank === 10) est += 0.18;
    else est += 0.12;
  }
  const bySuit = {};
  for (const c of offs) (bySuit[c.suit] = bySuit[c.suit] || []).push(c);
  for (const c of offs) {
    if (c.rank === 14) est += 0.50;
    else if (c.rank === 13) est += 0.16;
    else est += 0.03;
  }
  let shortBonus = 0.0;
  for (const s of nontrumpSuits(trump)) {
    const k = (bySuit[s] || []).length;
    if (k === 0) shortBonus += 0.35;
    else if (k === 1) shortBonus += 0.15;
  }
  shortBonus = Math.min(shortBonus, Math.max(0, nTrump - 1) * 0.40);
  est += shortBonus;
  return est;
}

function chooseDiscard(hand6, trump) {
  const nonTrump = hand6.filter((c) => !isTrump(c, trump));
  if (nonTrump.length === 0) {
    return hand6.reduce((a, b) => (cardValue(a, trump) <= cardValue(b, trump) ? a : b));
  }
  const suitCounts = {};
  for (const c of nonTrump) suitCounts[c.suit] = (suitCounts[c.suit] || 0) + 1;
  const voidMakers = nonTrump.filter((c) => suitCounts[c.suit] === 1 && c.rank <= 12);
  const pool = voidMakers.length ? voidMakers : nonTrump;
  return pool.reduce((a, b) => (cardValue(a, trump) <= cardValue(b, trump) ? a : b));
}

// ---------------------------------------------------------------------------
// bidding.py ports: loner_value_estimate, bid_candidates, net_bid_action
// ---------------------------------------------------------------------------

function infoStateFromGame(g, seat, knownDeadExtra) {
  const completed = g.trick_history.map((th) => th.plays);
  return {
    seat, phase: g.phase, dealer: g.dealer, target_score: g.target_score,
    hand: g.hands[seat].slice(), upcard: g.upcard, turned_down_suit: g.turned_down_suit,
    maker: g.maker, alone: g.alone, sitting_out: g.sitting_out, trump_settled: g.trump,
    current_player: g.current_player, leader: g.leader, current_trick: g.current_trick.slice(),
    completed_tricks: completed, played: g.played_cards.slice(), tricks_won: g.tricks_won.slice(),
    scores: g.scores.slice(), known_dead_extra: knownDeadExtra || [],
    bids_seen: g.bidsSeen,
  };
}

function lonerValueEstimate(lonerNet, g, seat, candidate) {
  let hand = g.hands[seat].slice();
  const round1Upcard = g.phase === PHASE_BID1 && candidate === g.upcard.suit;
  const turnedDown = round1Upcard ? null : g.upcard.suit;
  if (round1Upcard && seat === g.dealer) {
    const h6 = hand.concat([g.upcard]);
    const disc = chooseDiscard(h6, candidate);
    hand = h6.filter((c) => !cardEq(c, disc));
  }
  let leader = (g.dealer + 1) % 4;
  const sitting = partnerOf(seat);
  while (leader === sitting) leader = (leader + 1) % 4;
  const info = {
    seat, phase: PHASE_PLAY, dealer: g.dealer, target_score: g.target_score,
    hand, upcard: g.upcard, turned_down_suit: turnedDown, maker: seat, alone: true,
    sitting_out: sitting, trump_settled: candidate, current_player: leader, leader,
    current_trick: [], completed_tricks: [], played: [], tricks_won: [0, 0],
    scores: g.scores.slice(), known_dead_extra: [],
  };
  const x = encode(info, candidate);
  const { value } = lonerNet.forward(x);
  return value;
}

// Flat action index space -- mirrors euchre/actions.py exactly (the JS net's
// logits[59] are assembled in this same order, see PolicyValueNetJS.forward).
const ACTION_ = {
  PLAY_BASE: 0, DISCARD_BASE: 24, CALL_BASE: 48, CALL_ALONE_BASE: 52,
  ORDER_UP: 56, ORDER_UP_ALONE: 57, PASS: 58,
};
const NUM_ACTIONS_ = 59;

function bidCandidates(g) {
  if (g.phase === PHASE_BID1) return [g.upcard.suit];
  return SUITS.split("").filter((s) => s !== g.turned_down_suit);
}

function decodeBidAction(idx) {
  if (idx === ACTION_.PASS) return ["pass"];
  if (idx === ACTION_.ORDER_UP) return ["order_up", false];
  if (idx === ACTION_.ORDER_UP_ALONE) return ["order_up", true];
  if (idx >= ACTION_.CALL_BASE && idx < ACTION_.CALL_ALONE_BASE) {
    return ["call", SUITS[idx - ACTION_.CALL_BASE], false];
  }
  if (idx >= ACTION_.CALL_ALONE_BASE && idx < ACTION_.ORDER_UP) {
    return ["call", SUITS[idx - ACTION_.CALL_ALONE_BASE], true];
  }
  throw new Error("bad bid action index " + idx);
}

// One unified net forward pass, legal-action-masked -- mirrors
// rebel/train_rebel.py's ReBeLNetAgent.act() (greedy: argmax over the
// masked policy, same as this app already did with its old per-suit nets).
// A single call scores every candidate suit/alone combination at once (the
// suit-agnostic architecture's make_trump scorer runs over all 4 suits in
// one forward pass), so there's no more per-candidate loop or separate
// loner-value net -- alone-vs-not is just another pair of action logits.
function netBidAction(net, g, seat) {
  const info = infoStateFromGame(g, seat);
  const obs = encodeObservation(info);
  const { logits, value } = net.forward(obs);
  const mask = new Array(NUM_ACTIONS_).fill(false);
  if (g.phase === PHASE_BID1) {
    mask[ACTION_.PASS] = true;
    mask[ACTION_.ORDER_UP] = true;
    mask[ACTION_.ORDER_UP_ALONE] = true;
  } else {
    if (seat !== g.dealer) mask[ACTION_.PASS] = true;  // stick-the-dealer
    for (let si = 0; si < SUITS.length; si++) {
      if (SUITS[si] === g.turned_down_suit) continue;
      mask[ACTION_.CALL_BASE + si] = true;
      mask[ACTION_.CALL_ALONE_BASE + si] = true;
    }
  }
  const probs = maskedSoftmax(logits, mask);
  const idx = argmaxMasked(logits, mask);
  const action = decodeBidAction(idx);
  return { action, passProb: mask[ACTION_.PASS] ? probs[ACTION_.PASS] : 0,
          chosenProb: probs[idx], value };
}

// Greedy card choice from the unified net -- same for loner and normal play
// (the observation already encodes alone/sitting_out, so one net covers
// both; the old app used two separate nets for this).
function netCardAction(net, g, seat) {
  const info = infoStateFromGame(g, seat);
  const obs = encodeObservation(info);
  const { logits } = net.forward(obs);
  const legal = legalPlays(g.hands[seat], g.current_trick.map(([, c]) => c), g.trump);
  const mask = new Array(NUM_ACTIONS_).fill(false);
  const idx2card = {};
  for (const c of legal) {
    const idx = ACTION_.PLAY_BASE + cardIdOf(c);
    mask[idx] = true; idx2card[idx] = c;
  }
  const idx = argmaxMasked(logits, mask);
  return idx2card[idx];
}

function netDiscard(net, g) {
  const seat = g.dealer;
  const info = infoStateFromGame(g, seat);
  const obs = encodeObservation(info);
  const { logits } = net.forward(obs);
  const mask = new Array(NUM_ACTIONS_).fill(false);
  const idx2card = {};
  for (const c of g.hands[seat]) {
    const idx = ACTION_.DISCARD_BASE + cardIdOf(c);
    mask[idx] = true; idx2card[idx] = c;
  }
  const idx = argmaxMasked(logits, mask);
  return idx2card[idx];
}

const EuchreGameJS = {
  PHASE_BID1, PHASE_BID2, PHASE_DISCARD, PHASE_PLAY, PHASE_HAND_DONE, PHASE_GAME_DONE,
  ALONE_VALUE_TH, SimpleRNG, EuchreGame,
  estimateTricks, chooseDiscard, cardValue, netDiscard,
  infoStateFromGame, lonerValueEstimate, bidCandidates, netBidAction, netCardAction,
};
if (typeof window !== "undefined") Object.assign(window, EuchreGameJS);
if (typeof module !== "undefined") module.exports = EuchreGameJS;
