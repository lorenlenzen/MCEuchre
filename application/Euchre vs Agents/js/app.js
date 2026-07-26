// app_v4.js — you play South against three trained-agent seats, random deals,
// full games to 10. Every completed hand is recorded as an exact action
// transcript; clicking it in the history list replays it with all four hands
// exposed, stepped via Back/Forward buttons. Replay never recomputes a
// decision -- it deterministically re-applies the recorded actions to a fresh
// game, so it can never show anything other than exactly what happened.

const SEAT_NAME = { 0: "You", 1: "West", 2: "North", 3: "East" };
const SUIT_SYM = { C: "♣", D: "♦", H: "♥", S: "♠" };
const SUIT_NAME = { C: "Clubs", D: "Diamonds", H: "Hearts", S: "Spades" };
const SUIT_COLOR = { C: "black", D: "red", H: "red", S: "black" };
const RANK_LABEL = { 9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A" };
const YOU = 0;

// Display-only sort for rendering a hand: suit groups (fixed order, matching
// the deck-picker convention established elsewhere), then descending rank
// within each suit. Never mutates game.hands itself -- only the on-screen
// presentation, so nothing that indexes into the real hand array is affected.
const DISPLAY_SUIT_ORDER = ["C", "H", "S", "D"];
function sortHandForDisplay(hand) {
  return hand.slice().sort((a, b) => {
    const suitDiff = DISPLAY_SUIT_ORDER.indexOf(a.suit) - DISPLAY_SUIT_ORDER.indexOf(b.suit);
    if (suitDiff !== 0) return suitDiff;
    return b.rank - a.rank;
  });
}

let policyNet;

// ---- live game state --------------------------------------------------
let game = null;
let awaitingUser = null;
let bid2Selection = null;
let aiTimer = null;
let lastTrickCount = 0;
let roundPlays = { 0: null, 1: null, 2: null, 3: null };
let roundTrickIndex = 0;
let liveLogLines = [];
let currentHandRecord = null;   // being recorded live
let handHistory = [];            // completed hands, persists across games

// ---- view state ---------------------------------------------------------
let viewMode = "live";           // 'live' | 'replay'
let replayHandIndex = null;
let replayStep = 0;

// ---- expert-correction state ---------------------------------------------
let corrections = [];          // accumulated {..InfoState fields, agent_action, corrected_action}
let flaggingStepIdx = null;    // transcript index currently being corrected (replay view), or null
let flagBid2Selection = null;

function initNets() {
  policyNet = new PolicyValueNetJS(WEIGHTS_V2);
}

function cloneCard(c) { return c ? { rank: c.rank, suit: c.suit } : c; }

// ---------------------------------------------------------------------------
// Logging (kept separate for live vs replay so switching views never mixes them)
// ---------------------------------------------------------------------------

// rec/stepIdx (optional): if given, this line gets an inline "flag this
// decision" button -- used for agent decisions only (see aiStep), so a
// player can correct an agent's call without leaving the live game, even
// while it's currently the player's own turn to act.
function appendLogLine(msg, cls) {
  const el = document.getElementById("log");
  const d = document.createElement("div");
  d.className = "entry" + (cls ? " " + cls : "");
  d.textContent = msg;
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
}
function renderLogFromLines(lines) {
  const el = document.getElementById("log");
  el.innerHTML = "";
  for (const { msg, cls } of lines) appendLogLine(msg, cls);
}
function log(msg, cls) {
  liveLogLines.push({ msg, cls });
  appendLogLine(msg, cls);
}

// ---------------------------------------------------------------------------
// Card rendering
// ---------------------------------------------------------------------------

function cardEl(card, opts) {
  opts = opts || {};
  const d = document.createElement("div");
  d.className = "card" + (opts.small ? " small" : "") + " " + SUIT_COLOR[card.suit];
  if (opts.clickable) d.classList.add("clickable");
  if (opts.illegal) d.classList.add("illegal");
  d.innerHTML = `<div class="rank">${RANK_LABEL[card.rank]}</div><div class="suit">${SUIT_SYM[card.suit]}</div>`;
  if (opts.onClick) d.addEventListener("click", opts.onClick);
  return d;
}
function backEl() {
  const d = document.createElement("div");
  d.className = "card back";
  return d;
}

// ---------------------------------------------------------------------------
// Round-play tracking (live) -- same "lingers until next round" pattern
// ---------------------------------------------------------------------------

function recordRoundPlay(seat, card) {
  if (game.trick_history.length !== roundTrickIndex) {
    roundPlays = { 0: null, 1: null, 2: null, 3: null };
    roundTrickIndex = game.trick_history.length;
  }
  roundPlays[seat] = card;
}
function clearRoundPlays() {
  roundPlays = { 0: null, 1: null, 2: null, 3: null };
  roundTrickIndex = 0;
}

// ---------------------------------------------------------------------------
// Starting a live game / hand, with transcript recording
// ---------------------------------------------------------------------------

function startLiveGame() {
  clearTimeout(aiTimer);
  clearTimeout(trickClearTimer);
  const seed = (Date.now() ^ (Math.random() * 1e9)) >>> 0;
  game = new EuchreGame(10, Math.floor(Math.random() * 4), new SimpleRNG(seed));
  liveLogLines = [];
  clearLogPanel();
  beginRecordingHand();
  log(`New game. ${SEAT_NAME[game.dealer]} deals first.`, "agent");
  viewMode = "live";
  render();
  scheduleNext();
}

function beginRecordingHand() {
  currentHandRecord = {
    dealer: game.dealer,
    hands: {
      0: game.hands[0].map(cloneCard), 1: game.hands[1].map(cloneCard),
      2: game.hands[2].map(cloneCard), 3: game.hands[3].map(cloneCard),
    },
    upcard: cloneCard(game.upcard),
    scoresBefore: game.scores.slice(),
    transcript: [],
  };
  clearRoundPlays();
  lastTrickCount = 0;
}

function finalizeHandRecord() {
  currentHandRecord.result = { ...game.last_hand_result };
  currentHandRecord.scoresAfter = game.scores.slice();
  handHistory.push(currentHandRecord);
  currentHandRecord = null;
}

function clearLogPanel() {
  document.getElementById("log").innerHTML = "";
}

// ---------------------------------------------------------------------------
// LIVE rendering
// ---------------------------------------------------------------------------

function seatLabelText(seat, g) {
  let t = SEAT_NAME[seat];
  if (g && g.maker === seat) t += g.alone ? " ★ alone" : " ★";
  return t;
}

function renderDealerBadges(dealerSeat) {
  for (let seat = 0; seat < 4; seat++) {
    const el = document.getElementById(`dealer-${seat}`);
    const isDealer = dealerSeat === seat;
    el.classList.toggle("dealer-on", isDealer);
    el.style.display = isDealer ? "" : "none";
  }
}

function renderTurnGlow(currentPlayer, phaseActive) {
  for (let seat = 0; seat < 4; seat++) {
    const isTurn = phaseActive && currentPlayer === seat;
    document.getElementById(`label-${seat}`).classList.toggle("turn", isTurn);
    document.getElementById(`seat-${seat}`).classList.toggle("active-turn", isTurn);
  }
}

function renderLive() {
  document.getElementById("history-title").textContent = "Hand History";
  document.getElementById("playback-controls").style.display = "none";
  document.getElementById("score-a").value = game.scores[0];
  document.getElementById("score-b").value = game.scores[1];
  renderDealerBadges(game.dealer);
  const phaseActive = !(game.phase === "hand_done" || game.done);
  renderTurnGlow(game.current_player, phaseActive);

  for (let seat = 0; seat < 4; seat++) {
    const handDiv = document.getElementById(`hand-${seat}`);
    handDiv.innerHTML = "";
    document.getElementById(`label-${seat}`).textContent = seatLabelText(seat, game);
    const tricks = game.tricks_won_by_seat[seat];
    document.getElementById(`tricks-${seat}`).textContent = tricks;
    if (seat === YOU) {
      const hand = game.hands[seat];
      const isTurn = awaitingUser && game.current_player === seat;
      let legal = [];
      if (isTurn && (awaitingUser.kind === "play" || awaitingUser.kind === "discard")) {
        legal = awaitingUser.kind === "discard" ? hand
          : legalPlays(hand, game.current_trick.map(([, c]) => c), game.trump);
      }
      for (const c of sortHandForDisplay(hand)) {
        const isLegal = isTurn && legal.some((lc) => cardEq(lc, c));
        handDiv.appendChild(cardEl(c, {
          clickable: isLegal,
          illegal: isTurn && awaitingUser.kind === "play" && !isLegal,
          onClick: isLegal ? () => onUserCard(c) : null,
        }));
      }
    } else {
      for (let i = 0; i < game.hands[seat].length; i++) handDiv.appendChild(backEl());
    }
    const playedDiv = document.getElementById(`played-${seat}`);
    playedDiv.innerHTML = "";
    if (roundPlays[seat]) playedDiv.appendChild(cardEl(roundPlays[seat], {}));
    else {
      const ph = document.createElement("div");
      ph.className = "played-slot-empty";
      playedDiv.appendChild(ph);
    }
  }
  renderCenter(game);
  renderLiveActionBar();
  renderHistoryList();
}

function renderCenter(g) {
  const el = document.getElementById("center-area");
  el.innerHTML = "";
  if (g.trump) {
    const badge = document.createElement("div");
    badge.className = "trump-badge";
    badge.innerHTML = `${SUIT_SYM[g.trump]} Trump: ${SUIT_NAME[g.trump]}`;
    el.appendChild(badge);
  }
  if (g.phase === "bid1" || (g.phase === "bid2" && g.upcard)) {
    const wrap = document.createElement("div");
    wrap.className = "upcard-block";
    const cap = document.createElement("div");
    cap.className = "upcard-caption";
    cap.textContent = g.phase === "bid1" ? "Upcard" : "Turned down";
    wrap.appendChild(cap);
    const c = cardEl(g.upcard, {});
    if (g.phase === "bid2") c.style.opacity = "0.4";
    wrap.appendChild(c);
    el.appendChild(wrap);
  }
}

function clearActionBar() { document.getElementById("actionbar").innerHTML = ""; }
function addBtn(label, cls, onClick) {
  const b = document.createElement("button");
  b.className = "btn" + (cls ? " " + cls : "");
  b.textContent = label;
  b.addEventListener("click", onClick);
  document.getElementById("actionbar").appendChild(b);
  return b;
}

function renderLiveActionBar() {
  clearActionBar();
  document.getElementById("banner-slot").innerHTML = "";

  if (game.done || game.phase === "hand_done") {
    const r = game.last_hand_result;
    const slot = document.getElementById("banner-slot");
    const b = document.createElement("div");
    b.className = "banner";
    if (game.phase === "hand_done") {
      const title = r.euchred ? `Euchred! Defenders score ${r.points}.`
        : `${SEAT_NAME[r.maker]} made it${r.alone ? " alone" : ""}: +${r.points}.`;
      b.innerHTML = `<h2>${title}</h2><p>Tricks by the makers: ${r.maker_tricks} / 5</p>`;
      slot.appendChild(b);
      addBtn("Continue", "", nextLiveHand);
    } else {
      b.innerHTML = `<h2>Game over</h2><p>Final: ${game.scores[0]} – ${game.scores[1]}</p>`;
      slot.appendChild(b);
      addBtn("New Game", "", startLiveGame);
    }
    return;
  }
  if (!awaitingUser) return;

  const seat = game.current_player;
  if (awaitingUser.kind === "bid1") {
    addBtn("Order it up", "", () => submitBid(["order_up", false]));
    addBtn("Order it up, alone", "", () => submitBid(["order_up", true]));
    addBtn("Pass", "secondary", () => submitBid(["pass"]));
  } else if (awaitingUser.kind === "bid2") {
    const wrap = document.createElement("div");
    wrap.className = "suit-row";
    const suits = "CDHS".split("").filter((s) => s !== game.turned_down_suit);
    for (const s of suits) {
      const chip = document.createElement("div");
      chip.className = "suit-chip " + SUIT_COLOR[s] + (bid2Selection === s ? " selected" : "");
      chip.textContent = SUIT_SYM[s];
      chip.addEventListener("click", () => { bid2Selection = s; renderLiveActionBar(); });
      wrap.appendChild(chip);
    }
    document.getElementById("actionbar").appendChild(wrap);
    const isDealer = seat === game.dealer;
    const callBtn = addBtn(bid2Selection ? `Call ${SUIT_NAME[bid2Selection]}` : "Call…", "", () => {
      if (bid2Selection) submitBid(["call", bid2Selection, false]);
    });
    if (!bid2Selection) callBtn.disabled = true;
    const aloneBtn = addBtn("Call alone", "", () => { if (bid2Selection) submitBid(["call", bid2Selection, true]); });
    if (!bid2Selection) aloneBtn.disabled = true;
    if (!isDealer) addBtn("Pass", "secondary", () => submitBid(["pass"]));
  } else if (awaitingUser.kind === "discard") {
    const note = document.createElement("div");
    note.className = "setup-note";
    note.textContent = "Pick a card from your hand to discard.";
    document.getElementById("actionbar").appendChild(note);
  }
}

// ---------------------------------------------------------------------------
// Live user actions
// ---------------------------------------------------------------------------

function logBidAction(seat, action) {
  const who = SEAT_NAME[seat];
  const s = seat === YOU ? "" : "s";
  if (action[0] === "pass") return `${who} pass${s}.`;
  if (action[0] === "order_up") return `${who} order${s} it up${action[1] ? ", alone" : ""}.`;
  if (action[0] === "call") return `${who} call${s} ${SUIT_NAME[action[1]]}${action[2] ? ", alone" : ""}.`;
  return "";
}

function submitBid(action) {
  const seat = game.current_player;
  // Computed silently against the exact state the player just faced, before
  // it changes -- what the net would have done here. Not shown live; it's
  // only surfaced later if this hand gets reviewed in replay, where it can
  // be flagged/corrected as a training example (see renderFlagArea).
  const agentSuggestion = { type: "bid", action: netBidAction(policyNet, game, seat).action };
  const msg = logBidAction(seat, action);
  currentHandRecord.transcript.push({ type: "bid", seat, action, logLine: msg, agentSuggestion });
  log(msg, "you");
  awaitingUser = null;
  bid2Selection = null;
  game.step(action);
  postStep();
}

function onUserCard(card) {
  if (!awaitingUser) return;
  const seat = game.current_player;
  if (awaitingUser.kind === "discard") {
    const agentSuggestion = { type: "discard", card: cardStr(netDiscard(policyNet, game)) };
    const msg = `You discard ${RANK_LABEL[card.rank]}${SUIT_SYM[card.suit]}.`;
    currentHandRecord.transcript.push({ type: "discard", seat, card, logLine: msg, agentSuggestion });
    log(msg, "you");
    awaitingUser = null;
    game.step(["discard", card]);
    postStep();
  } else if (awaitingUser.kind === "play") {
    const agentSuggestion = { type: "play", card: cardStr(netCardAction(policyNet, game, seat)) };
    const msg = `You play ${RANK_LABEL[card.rank]}${SUIT_SYM[card.suit]}.`;
    recordRoundPlay(seat, card);
    currentHandRecord.transcript.push({ type: "play", seat, card, logLine: msg, agentSuggestion });
    log(msg, "you");
    awaitingUser = null;
    game.step(["play", card]);
    postStep();
  }
}

// ---------------------------------------------------------------------------
// Agent turns
// ---------------------------------------------------------------------------

function aiStep() {
  const seat = game.current_player;
  if (game.phase === "bid1" || game.phase === "bid2") {
    const info = netBidAction(policyNet, game, seat);
    const msg = logBidAction(seat, info.action);
    log(msg, "agent");
    log(`  (chosen_prob=${info.chosenProb.toFixed(2)}  value=${info.value.toFixed(2)})`, "agent");
    currentHandRecord.transcript.push({ type: "bid", seat, action: info.action, logLine: msg });
    game.step(info.action);
  } else if (game.phase === "discard") {
    const d = netDiscard(policyNet, game);
    const msg = `${SEAT_NAME[seat]} discards a card face-down.`;
    log(msg, "agent");
    currentHandRecord.transcript.push({ type: "discard", seat, card: d, logLine: msg });
    game.step(["discard", d]);
  } else if (game.phase === "play") {
    const card = netCardAction(policyNet, game, seat);
    const msg = `${SEAT_NAME[seat]} plays ${RANK_LABEL[card.rank]}${SUIT_SYM[card.suit]}.`;
    log(msg, "agent");
    recordRoundPlay(seat, card);
    currentHandRecord.transcript.push({ type: "play", seat, card, logLine: msg });
    game.step(["play", card]);
  }
  postStep();
}

// ---------------------------------------------------------------------------
// Step routing
// ---------------------------------------------------------------------------

let trickClearTimer = null;

function postStep() {
  const grew = game.trick_history.length > lastTrickCount;
  if (grew) {
    const last = game.trick_history[game.trick_history.length - 1];
    log(`${SEAT_NAME[last.winner]} wins the trick.`, last.winner === YOU ? "you" : "agent");
    lastTrickCount = game.trick_history.length;
    // Sweep the completed trick off the table after a short pause, regardless
    // of who leads next -- otherwise, if YOU won the trick, nothing forces a
    // clear until you choose your next lead, so the old cards can sit there
    // indefinitely while you're deciding (an agent winner only looked fine
    // because its own ~500ms auto-play delay happened to clear it soon after).
    const capturedTrickIndex = roundTrickIndex;
    clearTimeout(trickClearTimer);
    trickClearTimer = setTimeout(() => {
      if (roundTrickIndex === capturedTrickIndex) {   // no new round has started yet
        clearRoundPlays();
        render();
      }
    }, 1100);
  }
  if (game.phase === "hand_done" || game.done) {
    clearTimeout(trickClearTimer);
    clearRoundPlays();
    finalizeHandRecord();
  }
  render();
  if (game.phase === "hand_done" || game.done) return;
  scheduleNext();
}

function scheduleNext() {
  clearTimeout(aiTimer);
  if (game.phase === "hand_done" || game.done) { render(); return; }
  const seat = game.current_player;
  if (seat === YOU) {
    awaitingUser = { kind: game.phase };
    render();
  } else {
    awaitingUser = null;
    render();
    aiTimer = setTimeout(aiStep, game.phase === "play" ? 480 : 620);
  }
}

function nextLiveHand() {
  clearTimeout(aiTimer);
  clearTimeout(trickClearTimer);
  game.dealNextHand();
  beginRecordingHand();
  log(`— New hand. ${SEAT_NAME[game.dealer]} deals. —`, "agent");
  render();
  scheduleNext();
}

// ---------------------------------------------------------------------------
// Hand history list
// ---------------------------------------------------------------------------

function summarizeHand(rec) {
  const r = rec.result;
  const pts = r.euchred ? `defenders +${r.points}` : `+${r.points}${r.alone ? " (alone)" : ""}`;
  const trumpSym = r.trump ? SUIT_SYM[r.trump] : "";
  return `${SEAT_NAME[r.maker]} ${trumpSym} → ${pts}`;
}

function renderHistoryList() {
  const el = document.getElementById("history-list");
  el.innerHTML = "";
  if (handHistory.length === 0) {
    const empty = document.createElement("div");
    empty.className = "hist-empty";
    empty.textContent = "Completed hands will appear here for replay.";
    el.appendChild(empty);
    return;
  }
  for (let i = handHistory.length - 1; i >= 0; i--) {
    const rec = handHistory[i];
    const item = document.createElement("div");
    item.className = "hist-item" + (viewMode === "replay" && replayHandIndex === i ? " selected" : "");
    const summarySpan = document.createElement("span");
    summarySpan.innerHTML = `<span class="hist-num">#${i + 1}</span>${summarizeHand(rec)}`;
    summarySpan.style.cursor = "pointer";
    summarySpan.addEventListener("click", () => openReplay(i));
    const copyBtn = document.createElement("span");
    copyBtn.className = "hist-item-copy";
    copyBtn.textContent = "copy";
    copyBtn.addEventListener("click", (e) => { e.stopPropagation(); copyHandJSON(i); });
    item.appendChild(summarySpan);
    item.appendChild(copyBtn);
    item.addEventListener("click", () => openReplay(i));
    el.appendChild(item);
  }
}

// ---------------------------------------------------------------------------
// Replay engine: deterministic re-application of recorded actions
// ---------------------------------------------------------------------------

function buildReplayState(rec, k) {
  const g = new EuchreGame(10, rec.dealer, new SimpleRNG(1));
  g._startHand({
    hands: {
      0: rec.hands[0].map(cloneCard), 1: rec.hands[1].map(cloneCard),
      2: rec.hands[2].map(cloneCard), 3: rec.hands[3].map(cloneCard),
    },
    upcard: cloneCard(rec.upcard),
    dealer: rec.dealer,
  });
  g.scores = rec.scoresBefore.slice();
  let rp = { 0: null, 1: null, 2: null, 3: null };
  let rpTrickIdx = 0;
  for (let i = 0; i < k; i++) {
    const step = rec.transcript[i];
    if (step.type === "bid") {
      g.step(step.action);
    } else if (step.type === "discard") {
      g.step(["discard", step.card]);
    } else if (step.type === "play") {
      if (g.trick_history.length !== rpTrickIdx) {
        rp = { 0: null, 1: null, 2: null, 3: null };
        rpTrickIdx = g.trick_history.length;
      }
      rp[step.seat] = step.card;
      g.step(["play", step.card]);
    }
  }
  if (g.phase === "hand_done" || g.done) rp = { 0: null, 1: null, 2: null, 3: null };
  return { game: g, roundPlays: rp };
}

// What would the agent have done in YOUR seat, at the exact information state
// you actually faced at transcript step `idx`? Rebuilds that precise state
// (everything played before it, nothing after) and runs it through the SAME
// decision nets used for the other three seats -- a real counterfactual, not
// a guess, and it can never see anything you didn't legitimately see either
// (buildReplayState(rec, idx) only ever applies the first `idx` recorded
// actions, so hidden information from later in the hand isn't leaked in).
function describeBidActionForSuggestion(action) {
  if (action[0] === "pass") return "pass";
  if (action[0] === "order_up") return `order it up${action[1] ? ", alone" : ""}`;
  if (action[0] === "call") return `call ${SUIT_NAME[action[1]]}${action[2] ? ", alone" : ""}`;
  return "";
}

// Renders a stored agentSuggestion ({type:"bid",action} or {type:"play"|
// "discard",card: cardStr}) as display text. The suggestion itself is
// computed once, at the moment the player actually moves (see submitBid/
// onUserCard), against the exact pre-decision state -- not recomputed here
// or at replay time, so what's shown/flagged always matches what the net
// said in the moment, never a stale or re-derived value.
function describeAgentSuggestion(sugg) {
  if (!sugg) return null;
  if (sugg.type === "bid") return describeBidActionForSuggestion(sugg.action);
  const c = cardFromStr(sugg.card);
  const verb = sugg.type === "discard" ? "discard" : "play";
  return `${verb} ${RANK_LABEL[c.rank]}${SUIT_SYM[c.suit]}`;
}

// ---------------------------------------------------------------------------
// Expert-correction export: each record is the exact set of InfoState fields
// this project's Python encoder already consumes (encoder.py's InfoState),
// plus the agent's actual choice and the human's corrected label -- so this
// is directly loadable as a supervised fine-tuning example, not just a note.
// ---------------------------------------------------------------------------

function buildInfoStateRecord(rec, stepIdx) {
  const { game: g } = buildReplayState(rec, stepIdx);   // state BEFORE this step
  const step = rec.transcript[stepIdx];
  const seat = step.seat;
  return {
    seat, phase: g.phase, dealer: g.dealer, target_score: 10,
    hand: g.hands[seat].map(cardStr),
    upcard: cardStr(g.upcard),
    turned_down_suit: g.turned_down_suit,
    maker: g.maker, alone: g.alone, sitting_out: g.sitting_out,
    trump_settled: g.trump,
    candidate_trump: g.trump || g.upcard.suit,
    current_player: g.current_player, leader: g.leader,
    current_trick: g.current_trick.map(([p, c]) => [p, cardStr(c)]),
    completed_tricks: g.trick_history.map((th) => th.plays.map(([p, c]) => [p, cardStr(c)])),
    played: g.played_cards.map(([p, c]) => [p, cardStr(c)]),
    tricks_won: g.tricks_won.slice(),
    scores: g.scores.slice(),
  };
}

// The thing being corrected: for an agent's own step, that's just what it
// actually did. For the player's own step, there's no agent action to
// correct directly -- agentSuggestion (computed and stored at the moment
// the player moved, see submitBid/onUserCard) stands in for it, so a
// flagged correction here still means the same thing either way: "the net
// said X, the right answer is Y."
function actionRecordFromStep(step) {
  if (step.seat === YOU) return step.agentSuggestion;
  if (step.type === "bid") return { type: "bid", action: step.action };
  return { type: step.type, card: cardStr(step.card) };
}

function pushCorrection(rec, stepIdx, correctedAction) {
  const info = buildInfoStateRecord(rec, stepIdx);
  const step = rec.transcript[stepIdx];
  corrections.push({
    ...info,
    agent_action: actionRecordFromStep(step),
    corrected_action: correctedAction,
    hand_number: replayHandIndex + 1,
    step_index: stepIdx,
  });
  updateCorrectionsButton();
}

function submitCorrection(rec, stepIdx, correctedAction) {
  pushCorrection(rec, stepIdx, correctedAction);
  flaggingStepIdx = null;
  flagBid2Selection = null;
  render();
}

function startFlagging(stepIdx) {
  flaggingStepIdx = stepIdx;
  flagBid2Selection = null;
  render();
}
function cancelFlagging() {
  flaggingStepIdx = null;
  flagBid2Selection = null;
  render();
}


function renderFlagArea(rec) {
  const area = document.getElementById("flag-area");
  if (!area) return;
  area.innerHTML = "";
  if (replayStep === 0) return;
  const stepIdx = replayStep - 1;
  const step = rec.transcript[stepIdx];

  if (flaggingStepIdx !== stepIdx) {
    const btn = document.createElement("button");
    btn.className = "btn secondary";
    btn.style.width = "100%";
    btn.textContent = "⚑ Flag this decision";
    btn.addEventListener("click", () => startFlagging(stepIdx));
    area.appendChild(btn);
    return;
  }

  const { game: g } = buildReplayState(rec, stepIdx);   // exact state before the flagged step
  const label = document.createElement("div");
  label.className = "setup-note";
  label.textContent = step.seat === YOU
    ? "What's the correct play here?"
    : `What should ${SEAT_NAME[step.seat]} have done instead?`;
  area.appendChild(label);

  if (step.type === "play" || step.type === "discard") {
    const wrap = document.createElement("div");
    wrap.className = "hand-row";
    wrap.style.flexWrap = "wrap";
    for (const c of sortHandForDisplay(g.hands[step.seat])) {
      wrap.appendChild(cardEl(c, {
        clickable: true,
        onClick: () => submitCorrection(rec, stepIdx, { type: step.type, card: cardStr(c) }),
      }));
    }
    area.appendChild(wrap);
  } else if (step.type === "bid") {
    const acts = g.legalActions();
    const passAct = acts.find((a) => a[0] === "pass");
    const orderActs = acts.filter((a) => a[0] === "order_up");
    const callActs = acts.filter((a) => a[0] === "call");

    const btnRow = document.createElement("div");
    btnRow.className = "actionbar";
    if (passAct) {
      const b = document.createElement("button");
      b.className = "btn secondary"; b.textContent = "Pass";
      b.addEventListener("click", () => submitCorrection(rec, stepIdx, { type: "bid", action: ["pass"] }));
      btnRow.appendChild(b);
    }
    for (const oa of orderActs) {
      const b = document.createElement("button");
      b.className = "btn";
      b.textContent = oa[1] ? "Order up, alone" : "Order up";
      b.addEventListener("click", () => submitCorrection(rec, stepIdx, { type: "bid", action: oa }));
      btnRow.appendChild(b);
    }
    area.appendChild(btnRow);

    if (callActs.length) {
      const suits = [...new Set(callActs.map((a) => a[1]))];
      const chipRow = document.createElement("div");
      chipRow.className = "suit-row";
      for (const s of suits) {
        const chip = document.createElement("div");
        chip.className = "suit-chip " + SUIT_COLOR[s] + (flagBid2Selection === s ? " selected" : "");
        chip.textContent = SUIT_SYM[s];
        chip.addEventListener("click", () => { flagBid2Selection = s; render(); });
        chipRow.appendChild(chip);
      }
      area.appendChild(chipRow);
      const callBtnRow = document.createElement("div");
      callBtnRow.className = "actionbar";
      const cBtn = document.createElement("button");
      cBtn.className = "btn";
      cBtn.textContent = flagBid2Selection ? `Call ${SUIT_NAME[flagBid2Selection]}` : "Call…";
      cBtn.disabled = !flagBid2Selection;
      cBtn.addEventListener("click", () => submitCorrection(rec, stepIdx, { type: "bid", action: ["call", flagBid2Selection, false] }));
      const aBtn = document.createElement("button");
      aBtn.className = "btn";
      aBtn.textContent = "Call alone";
      aBtn.disabled = !flagBid2Selection;
      aBtn.addEventListener("click", () => submitCorrection(rec, stepIdx, { type: "bid", action: ["call", flagBid2Selection, true] }));
      callBtnRow.appendChild(cBtn);
      callBtnRow.appendChild(aBtn);
      area.appendChild(callBtnRow);
    }
  }

  const cancelBtn = document.createElement("button");
  cancelBtn.className = "btn secondary";
  cancelBtn.style.width = "100%";
  cancelBtn.style.marginTop = "6px";
  cancelBtn.textContent = "Cancel";
  cancelBtn.addEventListener("click", cancelFlagging);
  area.appendChild(cancelBtn);
}

// ---------------------------------------------------------------------------
// Clipboard copy, with a real fallback chain -- this file is often opened as
// a local file:// document, where the async Clipboard API is frequently
// unavailable, so a silent failure there would make the whole feature
// unreliable. Falls back to the older execCommand('copy') trick, and if even
// that isn't available, shows the JSON in a selected textarea so the person
// can always get the data out by hand.
// ---------------------------------------------------------------------------

function copyToClipboard(text) {
  if (typeof navigator !== "undefined" && navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(() => copyViaExecCommand(text));
  } else {
    copyViaExecCommand(text);
  }
}
function copyViaExecCommand(text) {
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand && document.execCommand("copy");
    document.body.removeChild(ta);
    if (!ok) showCopyFallbackModal(text);
  } catch (e) {
    showCopyFallbackModal(text);
  }
}
function showCopyFallbackModal(text) {
  const overlay = document.createElement("div");
  overlay.className = "copy-fallback-overlay";
  const box = document.createElement("div");
  box.className = "copy-fallback-box";
  const p = document.createElement("p");
  p.textContent = "Clipboard access isn't available here — select all and copy manually:";
  const ta = document.createElement("textarea");
  ta.value = text;
  const closeBtn = document.createElement("button");
  closeBtn.className = "btn";
  closeBtn.textContent = "Close";
  closeBtn.style.marginTop = "10px";
  closeBtn.addEventListener("click", () => overlay.remove());
  box.appendChild(p);
  box.appendChild(ta);
  box.appendChild(closeBtn);
  overlay.appendChild(box);
  document.body.appendChild(overlay);
  if (ta.select) { ta.focus(); ta.select(); }
}

function updateCorrectionsButton() {
  const btn = document.getElementById("copy-corrections-btn");
  if (btn) btn.textContent = `📋 Copy Corrections (${corrections.length})`;
}

function copyAllCorrections() {
  copyToClipboard(JSON.stringify(corrections, null, 2));
}

function copyHandJSON(idx) {
  const rec = handHistory[idx];
  const dump = {
    hand_number: idx + 1,
    dealer: rec.dealer,
    hands: {
      0: rec.hands[0].map(cardStr), 1: rec.hands[1].map(cardStr),
      2: rec.hands[2].map(cardStr), 3: rec.hands[3].map(cardStr),
    },
    upcard: cardStr(rec.upcard),
    scores_before: rec.scoresBefore,
    scores_after: rec.scoresAfter,
    result: rec.result,
    transcript: rec.transcript.map((t) => ({
      type: t.type, seat: t.seat, log: t.logLine,
      ...(t.card ? { card: cardStr(t.card) } : {}),
      ...(t.action ? { action: t.action } : {}),
    })),
  };
  copyToClipboard(JSON.stringify(dump, null, 2));
}

function openReplay(idx) {
  clearTimeout(aiTimer);
  viewMode = "replay";
  replayHandIndex = idx;
  replayStep = 0;
  render();
}

function returnToLive() {
  viewMode = "live";
  render();
  if (game && game.phase !== "hand_done" && !game.done) scheduleNext();
  else renderLogFromLines(liveLogLines);
}

function replayBack() {
  if (replayStep > 0) { replayStep--; render(); }
}
function replayForward() {
  const rec = handHistory[replayHandIndex];
  if (replayStep < rec.transcript.length) { replayStep++; render(); }
}

function renderReplay() {
  const rec = handHistory[replayHandIndex];
  const { game: rg, roundPlays: rrp } = buildReplayState(rec, replayStep);
  const atEnd = replayStep >= rec.transcript.length;

  document.getElementById("history-title").textContent = `Replaying hand #${replayHandIndex + 1}`;
  document.getElementById("playback-controls").style.display = "block";
  document.getElementById("pb-back").disabled = replayStep === 0;
  document.getElementById("pb-fwd").disabled = atEnd;
  const status = document.getElementById("pb-status");
  if (replayStep === 0) status.textContent = "Start of hand — click Forward to step through.";
  else status.textContent = `Step ${replayStep}/${rec.transcript.length}: ${rec.transcript[replayStep - 1].logLine}`;

  const scoresNow = atEnd ? rec.scoresAfter : rec.scoresBefore;
  document.getElementById("score-a").value = scoresNow[0];
  document.getElementById("score-b").value = scoresNow[1];
  renderDealerBadges(rec.dealer);
  renderTurnGlow(rg.current_player, !atEnd);

  for (let seat = 0; seat < 4; seat++) {
    const handDiv = document.getElementById(`hand-${seat}`);
    handDiv.innerHTML = "";
    document.getElementById(`label-${seat}`).textContent = seatLabelText(seat, atEnd ? null : rg);
    const tricks = rg.tricks_won_by_seat[seat];
    document.getElementById(`tricks-${seat}`).textContent = tricks;
    for (const c of sortHandForDisplay(rg.hands[seat])) handDiv.appendChild(cardEl(c, {}));   // all hands revealed
    const playedDiv = document.getElementById(`played-${seat}`);
    playedDiv.innerHTML = "";
    if (rrp[seat]) playedDiv.appendChild(cardEl(rrp[seat], {}));
    else {
      const ph = document.createElement("div");
      ph.className = "played-slot-empty";
      playedDiv.appendChild(ph);
    }
  }
  renderCenter(rg);

  clearActionBar();
  document.getElementById("banner-slot").innerHTML = "";
  if (atEnd) {
    const r = rec.result;
    const b = document.createElement("div");
    b.className = "banner";
    const title = r.euchred ? `Euchred! Defenders score ${r.points}.`
      : `${SEAT_NAME[r.maker]} made it${r.alone ? " alone" : ""}: +${r.points}.`;
    b.innerHTML = `<h2>${title}</h2><p>Tricks by the makers: ${r.maker_tricks} / 5</p>`;
    document.getElementById("banner-slot").appendChild(b);
  }

  renderLogFromLines(buildReplayLogLines(rec, replayStep));
  renderFlagArea(rec);
  renderHistoryList();
}

function buildReplayLogLines(rec, uptoStep) {
  const lines = [];
  for (let i = 0; i < uptoStep; i++) {
    const t = rec.transcript[i];
    lines.push({ msg: t.logLine, cls: t.seat === YOU ? "you" : "agent" });
    if (t.seat === YOU && t.agentSuggestion) {
      lines.push({ msg: `   ↳ agent would ${describeAgentSuggestion(t.agentSuggestion)}`, cls: "agent" });
    }
  }
  return lines;
}

// ---------------------------------------------------------------------------
// Top-level render dispatch
// ---------------------------------------------------------------------------

function render() {
  if (viewMode === "live") renderLive();
  else renderReplay();
}

window.addEventListener("DOMContentLoaded", () => {
  initNets();
  document.getElementById("pb-back").addEventListener("click", replayBack);
  document.getElementById("pb-fwd").addEventListener("click", replayForward);
  document.getElementById("pb-live").addEventListener("click", returnToLive);
  document.getElementById("copy-corrections-btn").addEventListener("click", copyAllCorrections);
  updateCorrectionsButton();
  startLiveGame();
});

// debug hooks for headless testing
window.__getState = () => ({
  viewMode, game, awaitingUser, handHistory, replayHandIndex, replayStep,
  corrections, flaggingStepIdx, flagBid2Selection, currentHandRecord,
});
window.__submitBid = submitBid;
window.__onUserCard = onUserCard;
window.__nextLiveHand = nextLiveHand;
window.__startLiveGame = startLiveGame;
window.__openReplay = openReplay;
window.__replayBack = replayBack;
window.__replayForward = replayForward;
window.__returnToLive = returnToLive;
window.__buildReplayState = buildReplayState;
window.__startFlagging = startFlagging;
window.__cancelFlagging = cancelFlagging;
window.__submitCorrection = submitCorrection;
window.__copyAllCorrections = copyAllCorrections;
window.__copyHandJSON = copyHandJSON;
window.__buildInfoStateRecord = buildInfoStateRecord;
