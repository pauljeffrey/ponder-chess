/* Ponder web UI. Rules/legality: chess.js. Moves for the AI: POST /api/move (FastAPI). */
(() => {
const API = (new URLSearchParams(location.search).get("api") || window.PONDER_API || "").replace(/\/+$/, "");
const GLYPH = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const $ = (id) => document.getElementById(id);
const el = { board: $("board"), moves: $("moves"), thoughts: $("thoughts"), status: $("status"), think: $("thinking"),
  promo: $("promo"), modal: $("modal"), toast: $("toast") };

let watching = false, paused = false, game, human = "w", flipped = false, selected = null, legal = [], last = null, busy = false, soundOn = true, token = 0;

// ---------- sound (synthesised, no assets) ----------
let actx;
function tone(freq, dur, type = "triangle", vol = 0.15, slide = 0) {
  if (!soundOn) return;
  actx = actx || new (window.AudioContext || window.webkitAudioContext)();
  const o = actx.createOscillator(), g = actx.createGain(), t = actx.currentTime;
  o.type = type; o.frequency.setValueAtTime(freq, t);
  if (slide) o.frequency.exponentialRampToValueAtTime(freq * slide, t + dur);
  g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(actx.destination); o.start(t); o.stop(t + dur);
}
const sfx = { move: () => tone(180, 0.09, "triangle", 0.2, 0.6), capture: () => { tone(120, 0.16, "sawtooth", 0.14, 0.5); tone(240, 0.06, "square", 0.08) },
  check: () => { tone(660, 0.12, "sine", 0.12); setTimeout(() => tone(880, 0.16, "sine", 0.12), 90) },
  end: () => [523, 659, 784].forEach((f, i) => setTimeout(() => tone(f, 0.35, "sine", 0.13), i * 140)) };

// ---------- helpers ----------
const files = "abcdefgh";
const sqName = (r, c) => files[c] + (8 - r);       // r=0 is rank 8
function toast(msg) { el.toast.textContent = msg; el.toast.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => (el.toast.hidden = true), 4000); }

// ---------- rendering ----------
function render(animate = null) {
  const prev = animate ? snapshot() : null;
  el.board.innerHTML = "";
  const board = game.board();
  for (let i = 0; i < 64; i++) {
    const r = flipped ? 7 - Math.floor(i / 8) : Math.floor(i / 8), c = flipped ? 7 - (i % 8) : i % 8;
    const name = sqName(r, c), cell = board[r][c];
    const d = document.createElement("div");
    d.className = "sq " + ((r + c) % 2 ? "d" : "l");
    d.dataset.sq = name;
    if (last && (last.from === name || last.to === name)) d.classList.add("last");
    if (selected === name) d.classList.add("sel");
    if (cell && cell.type === "k" && game.in_check() && cell.color === game.turn()) d.classList.add("check");
    if (i % 8 === 0) d.insertAdjacentHTML("beforeend", `<span class="coord r">${8 - r}</span>`);
    if (i >= 56) d.insertAdjacentHTML("beforeend", `<span class="coord f">${files[c]}</span>`);
    const mv = legal.find((m) => m.to === name);
    if (mv) d.insertAdjacentHTML("beforeend", `<span class="${cell ? "ring" : "dot"}"></span>`);
    if (cell) d.insertAdjacentHTML("beforeend", `<span class="piece ${cell.color}">${GLYPH[cell.type]}︎</span>`);
    el.board.appendChild(d);
  }
  if (animate && prev) slide(animate, prev);
  renderMoves(); renderCaptured(); syncButtons();
  el.status.textContent = game.game_over() ? "" : !watching && game.turn() === human ? "Your move" : "";
}
function snapshot() { const m = {}; el.board.querySelectorAll(".sq").forEach((s) => (m[s.dataset.sq] = s.getBoundingClientRect())); return m; }
function slide(mv, prev) {
  const to = el.board.querySelector(`[data-sq="${mv.to}"] .piece`), a = prev[mv.from], b = prev[mv.to];
  if (!to || !a) return;
  const nb = el.board.querySelector(`[data-sq="${mv.to}"]`).getBoundingClientRect();
  to.animate([{ transform: `translate(${a.left - nb.left}px,${a.top - nb.top}px)`, zIndex: 5 }, { transform: "none" }], { duration: 260, easing: "cubic-bezier(.2,.8,.2,1)" });
}
function renderMoves() {
  const h = game.history(); el.moves.innerHTML = "";
  for (let i = 0; i < h.length; i += 2) {
    el.moves.insertAdjacentHTML("beforeend", `<li class="n">${i / 2 + 1}.</li><li class="m ${i === h.length - 1 ? "cur" : ""}">${h[i]}</li><li class="m ${i + 1 === h.length - 1 ? "cur" : ""}">${h[i + 1] || ""}</li>`);
  }
  el.moves.scrollTop = el.moves.scrollHeight;
}
function renderCaptured() {
  const val = { p: 1, n: 3, b: 3, r: 5, q: 9 }, start = { p: 8, n: 2, b: 2, r: 2, q: 1 };
  const have = { w: { ...start }, b: { ...start } };
  game.board().flat().forEach((c) => c && c.type !== "k" && have[c.color][c.type]--);
  const lost = (col) => Object.keys(start).map((t) => GLYPH[t].repeat(have[col][t])).join("");
  const score = (col) => Object.keys(start).reduce((s, t) => s + val[t] * have[col][t], 0);
  const diff = score(human === "w" ? "b" : "w") - score(human);   // material the human has won
  $("botCaps").textContent = lost(human === "w" ? "b" : "w") + (diff > 0 ? "  +" + diff : "");
  $("topCaps").textContent = lost(human) + (diff < 0 ? "  +" + -diff : "");
}
function setEval(whiteScore) {
  const pct = Math.round(whiteScore * 100), p = flipped ? 100 - pct : pct;   // bar always shows the bottom player's side
  $("evalFill").style.height = (human === "w" ? pct : 100 - pct) + "%";
  $("evalText").textContent = (human === "w" ? pct : 100 - pct) + "%";
}
let lastRes = null, lastCtx = null, openIdx = null;
function showPartial(res, ctx) { lastRes = res; lastCtx = ctx; drawThoughts(); }   // variations growing, scores still to come
function showThoughts(res, ctx) { lastRes = res; lastCtx = ctx; openIdx = res.chosen; drawThoughts(); }
function drawThoughts() {
  const res = lastRes, ctx = lastCtx; if (!res) return;
  const who = ctx.turn === "w" ? "White" : "Black";
  $("thinkMs").textContent = res.ms == null ? "calculating…" : `${res.ms} ms`;
  const rootPct = Math.round((ctx.turn === "w" ? res.root_eval_white : 1 - res.root_eval_white) * 100);
  $("thinkMeta").innerHTML =
    `<span>${ctx.analysing ? "Analysis for" : "Move for"} <b>${who}</b></span><span><b>${res.candidates.length}</b> candidates</span>` +
    `${ctx.look === false ? "<span><b>first instinct</b> only</span>" : `<span>depth <b>${ctx.depth}</b></span>`}${ctx.look !== false && ctx.branch > 1 ? `<span><b>${ctx.branch}</b> replies each</span>` : ""}` +
    `<span>position looks <b>${rootPct}%</b> for ${who}</span>`;
  const fmtLine = (moves) => {
    let n = ctx.fullmove, t = ctx.turn;
    return moves.map((m, j) => {
      const num = t === "w" ? `${n}.` : j === 0 ? `${n}…` : "";
      const html = `${num ? `<i class="n">${num}</i>` : ""}<span class="mv ${j === 0 ? "first" : t === ctx.turn ? "me" : "opp"}">${m}</span>`;
      if (t === "b") n++; t = t === "w" ? "b" : "w";
      return html;
    }).join(" ");
  };
  const pctOf = (w) => (w == null ? null : Math.round((ctx.turn === "w" ? w : 1 - w) * 100));
  const pcTxt = (w) => (w == null ? "…" : pctOf(w) + "%");
  el.thoughts.classList.toggle("dense", res.candidates.length > 6);
  el.thoughts.innerHTML = res.candidates.map((c, i) => {
    const pct = pctOf(c.score_white) ?? 0, calc = c.score_white == null, prior = Math.round(c.prior * 100);
    const br = c.branches || [{ line: c.line, score_white: c.score_white, critical: true }];
    const many = br.length > 1, open = i === openIdx;
    const crit = br.find((b) => b.critical) || br[0];
    const body = open && many
      ? `<div class="bnote">${br.length} replies explored. Its worth is the reply that is worst for ${who} (highlighted).</div>` +
        br.map((b) => `<div class="bl ${b.critical ? "crit" : ""}"><div class="ln">${fmtLine(b.line)}</div><span class="bs">${pcTxt(b.score_white)}</span></div>`).join("")
      : `<div class="ln">${fmtLine(crit.line)}</div>`;
    return `<div class="cand ${i === res.chosen ? "pick" : ""} ${open ? "open" : ""} ${calc ? "calc" : ""}" data-i="${i}">
      <div class="h"><i class="rk">${i + 1}</i><b>${c.san}</b>${i === res.chosen ? '<em class="tag">chosen</em>' : ""}${many && !open ? `<span class="more">+${br.length - 1} replies</span>` : ""}
        <div class="mini" title="how good the position looks after this line"><i style="width:${pct}%"></i></div>
        <span class="prior" title="how much the AI liked this move at first glance">${prior}%</span><span class="pc">${calc ? "…" : pct + "%"}</span></div>
      ${body}</div>`;
  }).join("");
}
el.thoughts.addEventListener("click", (e) => {
  const c = e.target.closest(".cand"); if (!c) return;
  const i = +c.dataset.i; openIdx = openIdx === i ? null : i; drawThoughts();
});
async function callEngine(ctx, onPartial) {
  const r = await fetch(API + "/api/move/stream", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen: game.fen(), level: ctx.level, k: ctx.k, depth: ctx.depth, branch: ctx.branch, lookahead: ctx.look, model: modelId() }) });
  if (!r.ok) throw new Error((await r.json()).detail || r.status);
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "", result = null;
  for (;;) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    for (let i; (i = buf.indexOf("\n")) >= 0;) {
      const line = buf.slice(0, i); buf = buf.slice(i + 1); if (!line) continue;
      const ev = JSON.parse(line);
      if (ev.error) throw new Error(ev.error);
      if (ev.done) result = ev.result; else if (onPartial) onPartial(ev);
    }
  }
  if (!result) throw new Error("the server closed the connection");
  return result;
}
const searchCtx = (analysing) => ({ turn: game.turn(), fullmove: +game.fen().split(" ")[5], k: +$("kSlider").value,
  depth: +$("dSlider").value, branch: +$("bSlider").value, level: +$("level").value, look: $("lookahead").checked, analysing });
async function analyseNow() {
  if (busy || game.game_over()) return;
  busy = true; el.think.hidden = false; const my = ++token; const ctx = searchCtx(true);
  try {
    const res = await callEngine(ctx, (ev) => { if (my === token) showPartial(ev, ctx); });
    if (my === token) showThoughts(res, ctx);
  } catch (e) { toast("AI error: " + e.message); }
  finally { if (my === token) { busy = false; el.think.hidden = true; render(); } }
}
$("analyseBtn").onclick = analyseNow;

// ---------- game flow ----------
function newGame() {
  token++; busy = false; watching = $("mode").value === "watch"; paused = false; $("pauseBtn").textContent = "Pause";
  $("watchbox").hidden = !watching; $("sideLabel").textContent = watching ? "Ponder plays" : "Play as"; $("oppLabel").firstChild.textContent = watching ? "Ponder model" : "Opponent";
  const s = $("side").value; human = s === "r" ? (Math.random() < 0.5 ? "w" : "b") : s;
  game = new Chess(); redoStack = []; lastRes = null; selected = null; legal = []; last = null; flipped = human === "b";
  el.modal.hidden = true; el.promo.hidden = true; el.think.hidden = true;
  $("botAvatar").textContent = "♙"; $("topAvatar").textContent = watching ? "♜" : "♞"; setEval(0.5); setNames();
  el.thoughts.innerHTML = '<p class="muted">Candidate moves and the variations it calculated will appear here.</p>';
  render();
  if (watching) watchNext(); else if (game.turn() !== human) aiMove();
}
function afterMove(mv, who) {
  redoStack = [];
  last = { from: mv.from, to: mv.to }; selected = null; legal = [];
  render(mv);
  if (game.game_over()) return finish();
  (game.in_check() ? sfx.check : mv.captured ? sfx.capture : sfx.move)();
}
function finish() {
  sfx.end();
  let t, d;
  if (watching) {
    const ponderWon = game.in_checkmate() && game.turn() !== human;
    t = game.in_checkmate() ? (ponderWon ? "Ponder wins" : "Stockfish wins") : "Draw";
    d = game.in_checkmate() ? `Checkmate. Stockfish was ${sfLabel()}.` : game.in_stalemate() ? "Stalemate." : game.in_threefold_repetition() ? "Threefold repetition." : "Drawn position.";
  } else if (game.in_checkmate()) { const won = game.turn() !== human; t = won ? "You win!" : "Checkmate"; d = won ? "Brilliant. You beat the machine." : "The agent got you this time."; }
  else { t = "Draw"; d = game.in_stalemate() ? "Stalemate." : game.in_threefold_repetition() ? "Threefold repetition." : "Drawn position."; }
  $("modalTitle").textContent = t; $("modalText").textContent = d; el.modal.hidden = false;
}
// strength presets (k, depth); the sliders start from the chosen level and can then be set freely
const PRESET = { 1: [3, 1], 2: [3, 2], 3: [4, 4], 4: [4, 6], 5: [4, 8] };
function costNote() {
  if (!$("lookahead").checked) { $("cost").textContent = "no look-ahead · plays its first instinct · instant"; return; }
  const k = +$("kSlider").value, d = +$("dSlider").value, b = +$("bSlider").value, lines = k * b;
  const n = lines * d;                                    // moves the AI imagines in total
  const t = n < 400 ? "instant" : n < 1500 ? "a second or so" : n < 4000 ? "a few seconds" : "several seconds";
  $("cost").textContent = `${lines} line${lines > 1 ? "s" : ""} × ${d} moves ≈ ${n.toLocaleString()} imagined moves · ${t}` + (d > 8 ? " · beyond 8 moves it is out of its depth" : "");
}
function syncSliders(fromLevel) {
  if (fromLevel) { const [k, d] = PRESET[$("level").value]; $("kSlider").value = k; $("dSlider").value = d; $("bSlider").value = 1; }
  $("kVal").textContent = $("kSlider").value; $("dVal").textContent = $("dSlider").value; $("bVal").textContent = $("bSlider").value; costNote();
  const [pk, pd] = PRESET[$("level").value];
  $("cfgNote").textContent = (+$("kSlider").value === pk && +$("dSlider").value === pd && +$("bSlider").value === 1) ? "level preset" : "custom";
}
$("lookahead").addEventListener("change", () => {
  const on = $("lookahead").checked;
  ["dSlider", "bSlider"].forEach((id) => { $(id).disabled = !on; });
  syncSliders(false);
});
$("level").addEventListener("change", () => syncSliders(true));
$("kSlider").addEventListener("input", () => syncSliders(false));
$("dSlider").addEventListener("input", () => syncSliders(false));
$("bSlider").addEventListener("input", () => syncSliders(false));
syncSliders(true);

async function aiMove() {
  const my = ++token; busy = true; el.think.hidden = false; el.status.textContent = "";
  const ctx = searchCtx(false);
  try {
    const res = await callEngine(ctx, (ev) => { if (my === token) showPartial(ev, ctx); });
    if (my !== token) return;                       // game was reset / undone meanwhile
    const mv = game.move({ from: res.move.slice(0, 2), to: res.move.slice(2, 4), promotion: res.move[4] });
    if (!mv) throw new Error("server returned an illegal move: " + res.move);
    setEval(res.eval_white); showThoughts(res, ctx);
    afterMove(mv, "ai");
    if (watching) setTimeout(watchNext, 450);
  } catch (e) { toast("AI error: " + e.message); }
  finally { if (my === token) { busy = false; el.think.hidden = true; render(); } }
}
function tryMove(from, to) {
  const cands = game.moves({ verbose: true }).filter((m) => m.from === from && m.to === to);
  if (!cands.length) return false;
  if (cands.length > 1) { askPromotion(from, to); return true; }       // promotion
  commit({ from, to }); return true;
}
function commit(m) {
  const mv = game.move(m); if (!mv) return;
  afterMove(mv, "human");
  if (!game.game_over()) aiMove();
}
function askPromotion(from, to) {
  el.promo.innerHTML = "<div>" + ["q", "r", "b", "n"].map((p) => `<button data-p="${p}">${GLYPH[p]}︎</button>`).join("") + "</div>";
  el.promo.hidden = false;
  el.promo.onclick = (e) => { const b = e.target.closest("button"); if (!b) return; el.promo.hidden = true; commit({ from, to, promotion: b.dataset.p }); };
}

// ---------- input: click + drag ----------
let drag = null;
el.board.addEventListener("pointerdown", (e) => {
  if (busy || watching || game.game_over() || game.turn() !== human) return;
  const sq = e.target.closest(".sq"); if (!sq) return;
  const name = sq.dataset.sq;
  if (selected && legal.some((m) => m.to === name)) { if (tryMove(selected, name)) return; }
  const piece = game.get(name);
  if (piece && piece.color === human) {
    selected = name; legal = game.moves({ square: name, verbose: true }); render();
    const p = el.board.querySelector(`[data-sq="${name}"] .piece`);
    if (p) drag = { from: name, startX: e.clientX, startY: e.clientY, node: p, ghost: null };
  } else { selected = null; legal = []; render(); }
});
window.addEventListener("pointermove", (e) => {
  if (!drag) return;
  if (!drag.ghost && Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > 6) {
    const g = drag.node.cloneNode(true); g.classList.add("drag"); g.style.fontSize = getComputedStyle(drag.node).fontSize;
    document.body.appendChild(g); drag.ghost = g; drag.node.parentElement.classList.add("hide");
  }
  if (drag.ghost) { drag.ghost.style.left = e.clientX + "px"; drag.ghost.style.top = e.clientY + "px"; }
});
window.addEventListener("pointerup", (e) => {
  if (!drag) return;
  const d = drag; drag = null;
  if (!d.ghost) return;
  d.ghost.remove();
  const target = document.elementFromPoint(e.clientX, e.clientY)?.closest(".sq");
  if (target && target.dataset.sq !== d.from && tryMove(d.from, target.dataset.sq)) return;
  render();
});

// ---------- controls ----------
$("newBtn").onclick = newGame; $("modalBtn").onclick = newGame;
$("flipBtn").onclick = () => { flipped = !flipped; render(); };
$("tiltBtn").onclick = (e) => { document.body.classList.toggle("tilt"); e.target.classList.toggle("on"); };
$("soundBtn").onclick = (e) => { soundOn = !soundOn; e.target.classList.toggle("on", !soundOn); e.target.textContent = soundOn ? "Sound" : "Muted"; };
let redoStack = [];
function syncButtons() {
  $("undoBtn").disabled = watching || game.history().length === 0;
  $("redoBtn").disabled = watching || redoStack.length === 0;
  $("analyseBtn").disabled = watching;
}
$("undoBtn").onclick = () => {
  if (busy && !el.think.hidden) { token++; busy = false; el.think.hidden = true; }
  const n = game.turn() === human ? 2 : 1;
  for (let i = 0; i < n; i++) { const m = game.undo(); if (m) redoStack.push(m); }
  const h = game.history({ verbose: true }); last = h.length ? { from: h[h.length - 1].from, to: h[h.length - 1].to } : null;
  selected = null; legal = []; el.modal.hidden = true; render();
  if (game.turn() !== human) aiMove();
};
$("redoBtn").onclick = () => {
  if (busy || !redoStack.length) return;
  let m = redoStack.pop(), mv = game.move({ from: m.from, to: m.to, promotion: m.promotion });
  if (mv && redoStack.length && game.turn() !== human && !game.game_over()) {     // also replay the reply that followed it
    const r = redoStack.pop(); mv = game.move({ from: r.from, to: r.to, promotion: r.promotion }) || mv;
  }
  if (!mv) { redoStack = []; render(); return; }
  last = { from: mv.from, to: mv.to }; selected = null; legal = []; el.modal.hidden = true; render();
  if (game.game_over()) finish();
};
$("side").onchange = newGame;

// ---------- watch mode: Ponder vs Stockfish (Stockfish runs in the browser as WebAssembly) ----------
const sfLabel = () => ($("sfFull").checked ? "full strength" : `Elo ${$("sfElo").value}`);
function setNames() {
  $("botName").textContent = watching ? "Ponder" : "You";
  $("topName").textContent = watching ? `Stockfish · ${sfLabel()}` : "Ponder";
  if (watching) $("botCaps").textContent = "";
}
const sf = (() => {
  let w = null, ready = null, listeners = [];
  const send = (c) => w.postMessage(c);
  const on = (test, cb) => { const f = (l) => { if (test(l)) { listeners = listeners.filter((x) => x !== f); cb(l); } }; listeners.push(f); };
  function init() {
    if (ready) return ready;
    ready = new Promise((res, rej) => {
      try { w = new Worker("vendor/stockfish-19-lite-single.js"); } catch (e) { rej(e); return; }
      w.onmessage = (e) => listeners.slice().forEach((f) => f(String(e.data)));
      w.onerror = (e) => rej(new Error("Stockfish failed to load"));
      on((l) => l === "uciok", () => { send("isready"); on((l) => l === "readyok", res); });
      send("uci");
    });
    return ready;
  }
  async function best(fen, elo, ms) {
    await init();
    if (elo) { send("setoption name UCI_LimitStrength value true"); send("setoption name UCI_Elo value " + elo); }
    else send("setoption name UCI_LimitStrength value false");
    send("position fen " + fen);
    return new Promise((res) => {
      let score = null;
      const f = (l) => {
        const m = l.match(/score (cp|mate) (-?\d+)/);
        if (m) score = m[1] === "cp" ? +m[2] / 100 : (+m[2] > 0 ? 99 : -99);
        if (l.startsWith("bestmove")) { listeners = listeners.filter((x) => x !== f); res({ move: l.split(" ")[1], score }); }
      };
      listeners.push(f); send("go movetime " + ms);
    });
  }
  return { best, stop: () => w && send("stop") };
})();
function watchNext() {
  if (!watching || paused || busy || game.game_over()) return;
  if (game.turn() === human) aiMove(); else sfMove();
}
async function sfMove() {
  const my = ++token; busy = true; el.think.hidden = false; el.think.querySelector("em").textContent = "Stockfish thinking…";
  try {
    const turn = game.turn();
    const r = await sf.best(game.fen(), $("sfFull").checked ? 0 : +$("sfElo").value, +$("sfMs").value);
    if (my !== token) return;
    const mv = game.move({ from: r.move.slice(0, 2), to: r.move.slice(2, 4), promotion: r.move[4] });
    if (!mv) throw new Error("Stockfish returned an illegal move: " + r.move);
    afterMove(mv, "sf");
    setTimeout(watchNext, 450);
  } catch (e) { toast("Stockfish error: " + e.message); }
  finally { if (my === token) { busy = false; el.think.hidden = true; el.think.querySelector("em").textContent = "thinking…"; render(); } }
}
$("mode").onchange = newGame;
$("pauseBtn").onclick = () => { paused = !paused; $("pauseBtn").textContent = paused ? "Resume" : "Pause"; if (!paused) watchNext(); };
const sfUi = () => { $("sfEloVal").textContent = $("sfElo").value; $("sfMsVal").textContent = ($("sfMs").value / 1000).toFixed(1) + " s"; $("sfElo").disabled = $("sfFull").checked; setNames(); };
["sfElo", "sfMs", "sfFull"].forEach((id) => $(id).addEventListener("input", sfUi));
sfUi();

// ---------- model chooser ----------
let MODELS = [];
const modelId = () => $("model").value || undefined;
const modelLabel = () => (MODELS.find((m) => m.id === $("model").value) || { label: "Ponder" }).label;
function describeModel() {
  const m = MODELS.find((x) => x.id === $("model").value); if (!m) return;
  $("modelDesc").textContent = m.desc || "";
  $("modelBadge").textContent = !m.available ? "not ready yet" : !m.trained ? "untrained" :
    `${(m.params / 1e6).toFixed(0)}M parameters`;
  if (!watching) { $("topName").textContent = "Ponder"; $("topCaps").textContent = m.label; }
}
async function loadModels() {
  try {
    MODELS = await (await fetch(API + "/api/models")).json();
    const cur = $("model").value || (() => { try { return localStorage.getItem("model"); } catch (e) { return null; } })();
    $("model").innerHTML = MODELS.map((m) => `<option value="${m.id}" ${m.available ? "" : "disabled"}>${m.label}${m.available ? "" : " (not ready yet)"}</option>`).join("");
    const ok = MODELS.find((m) => m.id === cur && m.available) || MODELS.find((m) => m.available) || MODELS[0];
    if (ok) $("model").value = ok.id;
    describeModel();
  } catch (e) { /* server not reachable yet */ }
}
$("model").addEventListener("change", () => {
  try { localStorage.setItem("model", $("model").value); } catch (e) { /* private mode */ }
  describeModel(); toast(`Now playing against: ${modelLabel()}`);
});
loadModels(); setInterval(loadModels, 20000);   // a model that finishes training shows up without a page reload
newGame();
})();
