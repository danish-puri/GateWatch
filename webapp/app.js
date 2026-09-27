/* neoBird · Gate 1 Detection
 *
 * The design idea: a detection box with the corners taken off is a capsule,
 * so the interface and the data are made of the same shape. Four events get
 * four motions, and they are all defined here in one place.
 *
 * Data comes from DataSource. It ships in "mock" mode because the FastAPI
 * service (src/gcm_gatewatch/service/api.py) is still a stub. Flip MODE to
 * "live" once /stats, /crossings and /healthz are implemented and nothing
 * else in this file needs to change.
 */

'use strict';

const MODE = new URLSearchParams(location.search).get('mode') || 'mock';
const API_BASE = '';

/* ------------------------------------------------------------------ classes */

const CLASSES = {
  bus:    { label: 'Bus',       short: 'BUS',   color: '#1E4FE0', tint: '#6F97F1' },
  van:    { label: 'Van',       short: 'VAN',   color: '#7C5CE0', tint: '#B49CF2' },
  moto:   { label: 'Motorbike', short: 'MOTO',  color: '#2FB8C6', tint: '#7FDCE5' },
  cycle:  { label: 'Bicycle',   short: 'CYCLE', color: '#34C79A', tint: '#8CE3C7' },
  truck:  { label: 'Truck',     short: 'TRUCK', color: '#9AA3C2', tint: '#C7CDDF' },
  person: { label: 'Person',    short: 'PERSON',color: '#6B7392', tint: '#9AA3C2' },
};

/* Counted classes, in legend and equalizer order. People are detected and
   drawn but never counted: they are not the unit this gate cares about. */
const COUNTED = ['moto', 'bus', 'van', 'cycle', 'truck'];
const LEGEND  = ['bus', 'van', 'moto', 'cycle', 'person'];

/* --------------------------------------------------------------- detections
   Real YOLO v26 output over the clip, produced by tools/build_clip.py and
   tracked with the same supervision ByteTrack the service uses. Boxes are
   percentages of frame size, so the overlay fits the video element at any
   width without knowing the render size.

   The gate line is in the same 0-100 space as the boxes and as the <line> in
   the markup, so what is drawn is exactly what the crossing test measures. */

/* Placed from the tracks themselves, not by eye. Vehicles leaving this yard
   run upper-right to lower-left and pass under the camera, so the line lies
   across the bottom of their path. It is a SEGMENT, not an infinite line:
   testing against an infinite line counted parked buses at the far end of the
   yard every time their boxes wobbled. */
const GATE = { a: { x: 0, y: 66 }, b: { x: 66, y: 84 } };

/* Vehicle bottoms are clipped by the frame edge as they pass under the camera,
   which pins the foot point to y=100 and hides the crossing. The centroid does
   not saturate, so that is the reference point. */
const anchorOf = d => ({ x: d.x + d.w / 2, y: d.y + d.h / 2 });

/* Do the movement step and the gate segment actually intersect? Returns the
   side it came from, or 0 for no crossing. */
function crossesGate(p, q) {
  const cr = (o, u, v) => (u.x - o.x) * (v.y - o.y) - (u.y - o.y) * (v.x - o.x);
  const { a, b } = GATE;
  const d1 = cr(a, b, p), d2 = cr(a, b, q);
  const d3 = cr(p, q, a), d4 = cr(p, q, b);
  const straddles = (x, y) => (x > 0 && y < 0) || (x < 0 && y > 0);
  return straddles(d1, d2) && straddles(d3, d4) ? Math.sign(d1) : 0;
}

/* A box that jitters on the line would otherwise fire several times in a row. */
const CROSS_DEBOUNCE = 2.0;   // seconds

/* Only the largest vehicle carries a label. Twenty tags is noise, not data. */
const HERO_MIN_AREA = 8.0;    // percent of frame area

/* ------------------------------------------------------------------- state */

const state = {
  insideNow: 23,
  totalToday: 209,
  counts: { moto: 96, bus: 42, van: 38, cycle: 27, truck: 6 },
  byHour: [
    { hour: '06', v: 18 }, { hour: '08', v: 52 }, { hour: '10', v: 31 },
    { hour: '12', v: 24 }, { hour: '14', v: 46 }, { hour: 'now', v: 38 },
  ],
  crossings: [
    { ts: '16:20:41', cls: 'van',   dir: 'out', track: 4471, conf: 0.97, note: 'crossed the gate line', ago: 'just now', live: true },
    { ts: '16:19:08', cls: 'moto',  dir: 'in',  track: 4468, conf: 0.91, ago: '1 min ago' },
    { ts: '16:16:52', cls: 'bus',   dir: 'out', track: 4402, conf: 0.95, note: 'inside 3 h 12 min', ago: '4 min ago' },
    { ts: '16:14:20', cls: 'cycle', dir: 'in',  track: 4455, conf: 0.88, ago: '6 min ago' },
    { ts: '12:58:03', cls: 'bus',   dir: 'in',  track: 4310, conf: 0.93, flag: 'no exit yet · 3 h 22 min', ago: '3 h ago' },
  ],
  health: { ok: true, text: 'Both cameras healthy', lastWrite: 2 },
};

const MAX_ROWS = 6;

/* -------------------------------------------------------------- data source */

const DataSource = {
  async load() {
    if (MODE !== 'live') return null;
    const [stats, crossings, health] = await Promise.all([
      fetch(`${API_BASE}/stats`).then(r => r.json()),
      fetch(`${API_BASE}/crossings?limit=${MAX_ROWS}`).then(r => r.json()),
      fetch(`${API_BASE}/healthz`).then(r => r.json()),
    ]);
    return { stats, crossings, health };
  },

  /* Maps the service payload onto `state`. The API is the contract, so the
     shape below is the one api.py has to honour. */
  apply(payload) {
    if (!payload) return;
    const { stats, crossings, health } = payload;

    state.insideNow = stats.inside_now ?? state.insideNow;
    state.totalToday = stats.total_today ?? state.totalToday;
    if (stats.by_class) {
      for (const k of COUNTED) state.counts[k] = stats.by_class[k] ?? 0;
    }
    if (Array.isArray(stats.by_hour)) state.byHour = stats.by_hour;

    if (Array.isArray(crossings)) {
      state.crossings = crossings.slice(0, MAX_ROWS).map(c => ({
        ts: fmtClock(new Date(c.ts * 1000)),
        cls: normaliseClass(c.subject_type),
        dir: c.direction,
        track: c.track_id,
        conf: c.confidence ?? null,
        ago: relTime(c.ts),
      }));
      if (state.crossings[0]) state.crossings[0].live = true;
    }

    if (health) {
      state.health.ok = health.status === 'ok';
      state.health.text = health.status === 'ok' ? 'Both cameras healthy' : (health.detail || 'Camera problem');
      state.health.lastWrite = health.last_write_seconds ?? 0;
    }
  },
};

/* YOLO gives COCO names; the gate speaks in its own five. */
function normaliseClass(name) {
  const m = {
    bus: 'bus', truck: 'truck', car: 'van', van: 'van',
    motorcycle: 'moto', motorbike: 'moto', bicycle: 'cycle', person: 'person',
  };
  return m[String(name).toLowerCase()] || 'van';
}

/* ------------------------------------------------------------------- utils */

const $ = sel => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const pad = n => String(n).padStart(2, '0');
const fmtClock = d => `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;

function relTime(ts) {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (s < 45) return 'just now';
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  return `${Math.round(s / 3600)} h ago`;
}

const reduceMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/* --------------------------------------------------------------- rendering */

function renderLegend() {
  const host = $('#legend');
  host.replaceChildren();
  for (const key of LEGEND) {
    const c = CLASSES[key];
    const s = el('span', 'key' + (key === 'person' ? ' is-muted' : ''));
    const d = el('span', 'dot');
    d.style.background = c.color;
    s.append(d, document.createTextNode(c.label));
    host.append(s);
  }
}

/* --------------------------------------------------- overlay from the clip
   One DOM node per live track, reused frame to frame so the spring animation
   fires when a vehicle appears rather than on every repaint. */

const overlay = {
  data: null,        // parsed detections.json
  nodes: new Map(),  // track id -> element
  at: new Map(),     // track id -> last anchor point
  votes: new Map(),  // track id -> { class: summed confidence }
  lastCross: new Map(), // track id -> time of its last counted crossing
  cursor: -1,
};

/* Detection labels flip frame to frame (the same bus reads bus, then truck)
   while the track id stays put. So the class is decided per track by
   confidence-weighted vote, not by whatever the last frame happened to say. */
function voteClass(id, cls, conf) {
  let v = overlay.votes.get(id);
  if (!v) overlay.votes.set(id, (v = {}));
  v[cls] = (v[cls] || 0) + conf;
  let best = cls, bestScore = 0;
  for (const [k, s] of Object.entries(v)) if (s > bestScore) { best = k; bestScore = s; }
  return best;
}

async function loadDetections() {
  const res = await fetch('assets/gate-exit.detections.json');
  if (!res.ok) throw new Error(`detections ${res.status}`);
  overlay.data = await res.json();
  $('#cam-meta').textContent = `2880×1620 · 25 fps · YOLO v26 · ${overlay.data.tracks} tracks`;
}

/* Nearest detection frame at or before this playback time. */
function frameAt(t) {
  const f = overlay.data.frames;
  let lo = 0, hi = f.length - 1, best = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (f[mid].t <= t) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return best;
}

function paintFrame(t) {
  if (!overlay.data) return;
  const i = frameAt(t);
  if (i === overlay.cursor) return;
  overlay.cursor = i;

  const frame = overlay.data.frames[i];
  const dets = frame.d;
  const host = $('#dets');
  const alive = new Set();

  /* the biggest vehicle on screen carries the only label */
  let hero = null;
  for (const d of dets) {
    if (d.c === 'person') continue;
    const area = d.w * d.h;
    if (area >= HERO_MIN_AREA && (!hero || area > hero.w * hero.h)) hero = d;
  }

  for (const d of dets) {
    alive.add(d.i);
    const cls = d.c === 'person' ? 'person' : voteClass(d.i, d.c, d.p);
    const c = CLASSES[cls];
    let node = overlay.nodes.get(d.i);

    if (!node) {
      /* motion 01: a new track springs out of its own centroid */
      node = el('div', 'det' + (cls === 'person' ? ' is-person' : ''));
      host.append(node);
      overlay.nodes.set(d.i, node);
    }

    node.style.color = c.color;
    node.style.left = `${d.x}%`;
    node.style.top = `${d.y}%`;
    node.style.width = `${d.w}%`;
    node.style.height = `${d.h}%`;

    syncTag(node, d, cls, d === hero);
    testCrossing(d, cls, frame.t, node);
  }

  for (const [id, node] of overlay.nodes) {
    if (!alive.has(id)) {
      node.remove();
      overlay.nodes.delete(id);
      overlay.at.delete(id);
    }
  }
}

function syncTag(node, d, cls, isHero) {
  const has = node.firstElementChild;
  if (!isHero || cls === 'person') { if (has) has.remove(); node.classList.remove('is-hero'); return; }
  if (has) {
    has.querySelector('.conf').textContent = d.p.toFixed(2);
    has.querySelector('.short').textContent = CLASSES[cls].short;
    return;
  }
  node.append(buildTag(d, CLASSES[cls]));
  node.classList.add('is-hero');
}

function buildTag(d, c) {
  const tag = el('div', 'det-tag');

  const cls = el('div', 'cls');
  cls.style.backgroundImage = `linear-gradient(100deg, ${c.color}, ${c.tint})`;
  cls.append(el('span', 'short', c.short), el('span', 'conf', d.p.toFixed(2)));
  tag.append(cls);

  const trk = el('div', 'trk');
  trk.append(el('span', 'id', `#${d.i}`));
  tag.append(trk);
  return tag;
}

/* ------------------------------------------------------- crossing detection
   The only place a count is created. A vehicle counts when its movement step
   actually intersects the gate segment. */

function testCrossing(d, cls, t, node) {
  const cur = anchorOf(d);
  const prev = overlay.at.get(d.i);
  overlay.at.set(d.i, cur);
  if (!prev) return;

  const side = crossesGate(prev, cur);
  if (!side) return;
  if (cls === 'person') return;   // people are detected, never counted

  const last = overlay.lastCross.get(d.i);
  if (last !== undefined && t - last < CROSS_DEBOUNCE) return;
  overlay.lastCross.set(d.i, t);

  /* This is the exit camera, mounted above the gate looking into the yard.
     Vehicles that cross the line are driving under it and off the campus, so
     that direction is OUT. The reverse is a vehicle coming back in. */
  recordCrossing({ cls, track: d.i, conf: d.p, dir: side > 0 ? 'in' : 'out' }, node);
}

function recordCrossing(ev, node) {
  if (node) {
    node.classList.remove('is-crossing');
    void node.offsetWidth;
    node.classList.add('is-crossing');
    emitCount(node, ev.cls);          // motion 02
  }

  for (const r of state.crossings) {
    r.live = false;
    r.isNew = false;
    if (r.ago === 'just now') r.ago = '1 min ago';
  }
  state.crossings.unshift({
    ts: fmtClock(new Date()),
    cls: ev.cls,
    dir: ev.dir,
    track: ev.track,
    conf: ev.conf,
    ago: 'just now',
    live: true,
    isNew: true,
  });
  state.crossings = state.crossings.slice(0, MAX_ROWS);

  state.counts[ev.cls] = (state.counts[ev.cls] || 0) + 1;
  state.totalToday += 1;
  state.insideNow = Math.max(0, state.insideNow + (ev.dir === 'in' ? 1 : -1));
  state.byHour[state.byHour.length - 1].v += 1;

  renderRows();
  renderHero(true);
  renderEqualizer();                  // motion 03
  flashBar(ev.cls);
  renderDayFlow();
}

function renderEqualizer() {
  const host = $('#equalizer');
  const max = Math.max(...COUNTED.map(k => state.counts[k]), 1);
  const isPhone = window.matchMedia('(max-width: 720px)').matches;
  const track = isPhone ? 86 : 128;

  if (host.childElementCount !== COUNTED.length) {
    host.replaceChildren();
    for (const key of COUNTED) {
      const c = CLASSES[key];
      const bar = el('div', 'bar');
      bar.dataset.cls = key;
      const well = el('div', 'well');
      const cap = el('div', 'cap');
      cap.style.backgroundImage = `linear-gradient(180deg, ${c.color}, ${c.tint})`;
      well.append(cap);
      bar.append(well, el('div', 'n'), el('div', 'k', c.short));
      host.append(bar);
    }
  }

  for (const bar of host.children) {
    const key = bar.dataset.cls;
    const v = state.counts[key];
    /* min 16px so a class with almost nothing is still a capsule, not a line */
    bar.querySelector('.cap').style.height = `${Math.max(16, (v / max) * track)}px`;
    bar.querySelector('.n').textContent = v;
  }
}

function renderRows() {
  const host = $('#rows');
  host.replaceChildren();

  for (const r of state.crossings.slice(0, MAX_ROWS)) {
    host.append(buildRow(r));
  }
}

function buildRow(r) {
  const c = CLASSES[r.cls];
  const row = el('div', 'crow' + (r.live ? ' is-live' : '') + (r.isNew ? ' is-new' : ''));

  row.append(el('span', 't mono', r.ts));

  const dWrap = el('span', 'd');
  const solid = r.live && r.dir === 'out';
  const chip = el('span', `chip chip-${r.dir}` + (solid ? ' is-solid' : ''), r.dir.toUpperCase());
  dWrap.append(chip);
  row.append(dWrap);

  const cWrap = el('span', 'c');
  const dot = el('span', 'dot');
  dot.style.background = c.color;
  const meta = el('span', 'cmeta');
  meta.append(el('span', 'name', c.label));
  if (r.note) meta.append(el('span', 'note', r.note));
  if (r.flag) meta.append(el('span', 'flag', r.flag));
  cWrap.append(dot, meta);
  row.append(cWrap);

  row.append(el('span', 'id', `#${r.track}`));
  row.append(el('span', 'cf', r.conf == null ? '—' : r.conf.toFixed(2)));
  row.append(el('span', 'ago', r.ago));
  return row;
}

function renderHero(bump) {
  const node = $('#inside-now');
  node.textContent = state.insideNow;
  $('#total-today').textContent = state.totalToday;
  if (bump && !reduceMotion()) {
    node.classList.remove('is-bumped');
    void node.offsetWidth;
    node.classList.add('is-bumped');
  }
}

function renderDayFlow() {
  const host = $('#dotfield');
  const axis = $('#houraxis');
  const H = 180;
  const max = Math.max(...state.byHour.map(p => p.v), 1);
  const peak = state.byHour.reduce((a, b) => (b.v > a.v ? b : a), state.byHour[0]);

  host.replaceChildren();
  axis.replaceChildren();

  state.byHour.forEach((p, i) => {
    const ratio = p.v / max;
    const size = 18 + ratio * 18;
    const barH = ratio * 120;
    const isPeak = p === peak;
    const isNow = p.hour === 'now';

    const dot = el('div', 'fdot');
    dot.style.width = `${size}px`;
    dot.style.height = `${size}px`;
    dot.style.left = `${((i + 0.5) / state.byHour.length) * 100}%`;
    dot.style.top = `${H - barH - size}px`;
    /* one colour family: the dots read as volume, never as a class */
    dot.style.backgroundImage = isNow
      ? 'linear-gradient(180deg,#7C5CE0,#B49CF2)'
      : `linear-gradient(180deg, ${mixCobalt(ratio)}, ${mixCobalt(ratio * 0.4)})`;
    if (isPeak) dot.style.boxShadow = '0 10px 26px -8px rgba(30,79,224,.7)';
    host.append(dot);

    const t = el('span', isNow ? 'is-now' : isPeak ? 'is-peak' : null, p.hour);
    axis.append(t);
  });

  $('#peak-label').textContent = peak.hour === 'now' ? 'peak now' : `peak ${peak.hour}–${pad(Number(peak.hour) + 2)}`;
}

/* pale ice to deep ice, so a bigger dot is also a darker one */
function mixCobalt(t) {
  const a = [201, 217, 250], b = [30, 79, 224];
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * Math.min(1, Math.max(0, t))));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function renderHealth() {
  const h = state.health;
  $('#health-text').textContent = h.text;
  $('#health-dot').style.background = h.ok ? '#34C79A' : '#D4245E';
  $('#health-meta').textContent = `last write ${h.lastWrite}s ago`;

  const s = $('#status');
  s.classList.toggle('is-stale', h.ok && h.lastWrite > 20);
  s.classList.toggle('is-down', !h.ok);
  $('#status-label').textContent = h.ok ? (h.lastWrite > 20 ? 'Stale' : 'Live') : 'Camera down';
}

/* -------------------------------------------------------- motion 02: emit
   On crossing the capsule flashes, then throws a dot that flies into the
   crossings list. You watch the number get made instead of seeing it change. */

function emitCount(detNode, cls) {
  if (reduceMotion()) return;

  const from = detNode.getBoundingClientRect();
  const target = $('#rows').getBoundingClientRect();

  const dot = el('div', 'emit-dot');
  dot.style.color = CLASSES[cls].color;
  dot.style.position = 'fixed';
  dot.style.left = `${from.left + from.width / 2 - 7}px`;
  dot.style.top = `${from.top + from.height / 2 - 7}px`;
  dot.style.setProperty('--dx', `${target.left + 40 - (from.left + from.width / 2)}px`);
  dot.style.setProperty('--dy', `${target.top + 20 - (from.top + from.height / 2)}px`);
  dot.style.zIndex = '4';

  document.body.append(dot);
  dot.addEventListener('animationend', () => dot.remove(), { once: true });
}

/* --------------------------------------------------------- playback driver
   The clip is the clock. Boxes are repainted against video.currentTime, so
   the overlay stays locked to the footage even if playback stalls, seeks or
   the tab is backgrounded. */

function startPlayback(video) {
  const step = () => {
    paintFrame(video.currentTime);
    video.requestVideoFrameCallback
      ? video.requestVideoFrameCallback(step)
      : requestAnimationFrame(step);
  };
  video.requestVideoFrameCallback
    ? video.requestVideoFrameCallback(step)
    : requestAnimationFrame(step);

  /* On loop the same vehicles reappear on the far side of the line. Without
     clearing, every one of them would register a bogus crossing at the seam. */
  video.addEventListener('seeking', resetTracks);
  video.addEventListener('timeupdate', () => {
    if (video.currentTime < 0.4) resetTracks();
  });
}

function resetTracks() {
  overlay.cursor = -1;
  overlay.at.clear();
  overlay.votes.clear();
  overlay.lastCross.clear();
  for (const node of overlay.nodes.values()) node.remove();
  overlay.nodes.clear();
}

function flashBar(cls) {
  const bar = document.querySelector(`.bar[data-cls="${cls}"] .cap`);
  if (!bar) return;
  bar.classList.add('is-hit');
  setTimeout(() => bar.classList.remove('is-hit'), 600);
}

/* ------------------------------------------------------------------- clock */

function tickClock() {
  $('#clock').textContent = fmtClock(new Date());
}

function ageRows() {
  for (const r of state.crossings) {
    if (r.ago === 'just now' && !r.live) r.ago = '1 min ago';
  }
}

/* -------------------------------------------------------------------- boot */

async function refresh() {
  try {
    const payload = await DataSource.load();
    if (payload) {
      DataSource.apply(payload);
      state.health.ok = true;
    }
  } catch (err) {
    /* Never blank the screen on a failed poll. Freeze the content, mark it
       stale, and let the operator see that the data stopped rather than
       showing an empty dashboard that looks like zero traffic. */
    console.warn('poll failed', err);
    state.health.ok = false;
    state.health.text = 'Service unreachable';
  }
  renderAll();
}

function renderAll() {
  renderLegend();
  renderEqualizer();
  renderRows();
  renderHero(false);
  renderDayFlow();
  renderHealth();
}

async function init() {
  renderAll();
  tickClock();

  setInterval(tickClock, 1000);
  setInterval(() => {
    state.health.lastWrite = MODE === 'live' ? state.health.lastWrite + 2 : 2;
    renderHealth();
  }, 2000);

  window.addEventListener('resize', () => {
    renderEqualizer();
    renderDayFlow();
  });

  /* The clip and its detections are a pair. If either is missing the page
     still works: the poster frame stays up and the counts stop moving, which
     is the honest failure rather than an empty stage. */
  try {
    await loadDetections();
    const video = $('#feed');
    await video.play().catch(() => {});   // autoplay policy may refuse; poster remains
    startPlayback(video);
  } catch (err) {
    console.warn('clip unavailable, holding on the poster frame', err);
    $('#cam-meta').textContent = '2880×1620 · 25 fps · YOLO v26 · clip not loaded';
  }

  if (MODE === 'live') {
    refresh();
    setInterval(refresh, 3000);
  } else {
    setInterval(ageRows, 60000);
  }
}

document.addEventListener('DOMContentLoaded', init);

/* A single handle for debugging in the console and for the smoke test in
   tools/. Everything above stays module-private. */
window.neoBird = { state, overlay, CLASSES, GATE, paintFrame, recordCrossing };
