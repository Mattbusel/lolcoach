/* LoLCoach review UI.
 *
 * Plain JavaScript, no bundler and no network at runtime, so the packaged
 * desktop build works offline. Structure:
 *
 *   state      one object; every render reads from it, nothing else holds truth
 *   api        all fetch calls, so error handling lives in one place
 *   MapView    minimap: real Riot map, champion portraits, trails, heatmap
 *   GoldChart  team gold differential with objective and death markers
 *   WaveRibbon per-lane wave position over time
 *   render*    DOM writers, each owning exactly one region
 *
 * Every function is defined once. To change behaviour, edit the function
 * rather than appending a replacement.
 */

'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, html) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html !== undefined) node.innerHTML = html;
  return node;
};

const TEAM = { 100: '#4b9ce8', 200: '#e05c72' };
const GOLD = '#c8a961';
const KIND_COLOUR = {
  death: '#e05c72', objective: '#4fb391', recall: '#d9a441',
  wave: '#4b9ce8', rotation: '#9b7bd4',
};

const state = {
  status: null,
  matches: [],
  filter: '',
  match: null,
  stamps: [],
  index: 0,
  momentKind: 'all',
  playing: false,
  playTimer: null,
  activeMomentId: null,
  focusId: null,       // participant the review is centred on
  showTrails: true,
  onlyMine: true,
  showDeaths: false,
};

/* ------------------------------------------------------------------ utils */

function clock(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

function toast(message, isError = false) {
  const node = el('div', `toast${isError ? ' error' : ''}`, message);
  document.body.append(node);
  setTimeout(() => node.remove(), isError ? 6000 : 3200);
}

function crisp(canvas, cssHeight) {
  /* Size a canvas to its CSS box at device resolution so lines stay sharp. */
  const scale = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = cssHeight || canvas.clientHeight || 150;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

/* -------------------------------------------------------------------- api */

const api = {
  async get(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || response.statusText);
    return response.json();
  },
  /* ``timeoutMs`` guards interactive calls. Without it a slow upstream leaves
   * the UI waiting forever with its controls disabled, which is indent­ical to
   * a freeze from the user's side. */
  async post(path, body, timeoutMs = 0) {
    const controller = new AbortController();
    const timer = timeoutMs ? setTimeout(() => controller.abort(), timeoutMs) : null;
    let response;
    try {
      response = await fetch(path, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body), signal: controller.signal,
      });
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Riot did not answer in time. Nothing was saved — try again.');
      }
      throw error;
    } finally {
      if (timer) clearTimeout(timer);
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    return payload;
  },
};

/* ---------------------------------------------------------------- MapView */

/* The real Summoner's Rift minimap with champion portraits.
 *
 * Timeline coordinates have a bottom-left origin; canvas y grows downward, so
 * y is flipped. Portraits are cached per champion key; a missing portrait
 * degrades to an initials badge rather than vanishing. */
const MapView = {
  canvas: null, ctx: null, background: null, icons: new Map(),
  bounds: { min_x: -120, max_x: 14870, min_y: -120, max_y: 14980 },

  attach(canvas, bounds) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    if (bounds) this.bounds = bounds;
    const scale = window.devicePixelRatio || 1;
    const size = Math.round((canvas.clientWidth || 640) * scale);
    canvas.width = size;
    canvas.height = size;
    if (!this.background) {
      const image = new Image();
      image.onload = () => { this.background = image; this.draw(); };
      image.onerror = () => { this.background = null; this.draw(); };
      image.src = '/assets/map';
    }
  },

  icon(key) {
    if (!key) return null;
    if (this.icons.has(key)) return this.icons.get(key);
    const image = new Image();
    image.onload = () => this.draw();
    image.onerror = () => this.icons.set(key, null);
    image.src = `/assets/champion/${encodeURIComponent(key)}`;
    this.icons.set(key, image);
    return image;
  },

  project(x, y) {
    const b = this.bounds;
    const size = this.canvas.width;
    return [
      ((x - b.min_x) / (b.max_x - b.min_x)) * size,
      size - ((y - b.min_y) / (b.max_y - b.min_y)) * size,
    ];
  },

  draw() {
    if (!this.ctx || !this.canvas) return;
    const ctx = this.ctx;
    const size = this.canvas.width;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, size, size);

    if (this.background) {
      ctx.drawImage(this.background, 0, 0, size, size);
      ctx.fillStyle = 'rgba(5, 13, 22, 0.42)';
      ctx.fillRect(0, 0, size, size);
    } else {
      ctx.fillStyle = '#050d16';
      ctx.fillRect(0, 0, size, size);
      ctx.fillStyle = '#6b7d8f';
      ctx.font = `${Math.round(size / 38)}px system-ui`;
      ctx.textAlign = 'center';
      ctx.fillText('Minimap not cached yet', size / 2, size / 2);
      ctx.textAlign = 'start';
    }
    if (!state.match) return;

    if (state.showDeaths) this.drawDeaths(ctx, size);
    if (state.showTrails) this.drawTrails(ctx, size);
    this.drawChampions(ctx, size);
  },

  /* Where deaths happened, as a soft accumulation. Reading a whole game's
   * deaths at once is how you notice you keep dying in the same river. */
  drawDeaths(ctx, size) {
    const radius = size / 11;
    for (const moment of state.match.moments) {
      if (moment.kind !== 'death') continue;
      const { x, y, team_id: team } = moment.details || {};
      if (x === null || x === undefined) continue;
      if (state.focusId && moment.details.victim_id !== state.focusId) continue;
      const [px, py] = this.project(x, y);
      const glow = ctx.createRadialGradient(px, py, 0, px, py, radius);
      const base = team === 200 ? '224,92,114' : '75,156,232';
      glow.addColorStop(0, `rgba(${base},0.5)`);
      glow.addColorStop(1, `rgba(${base},0)`);
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  },

  /* Movement trail for the focused player over the preceding two minutes. */
  drawTrails(ctx, size) {
    if (!state.focusId) return;
    const now = state.stamps[state.index];
    const path = state.match.frames
      .filter((f) => f.participant_id === state.focusId && f.x !== null
        && f.ts_ms <= now && f.ts_ms >= now - 120_000)
      .sort((a, b) => a.ts_ms - b.ts_ms);
    if (path.length < 2) return;
    ctx.lineWidth = Math.max(2, size / 300);
    ctx.lineCap = 'round';
    for (let i = 1; i < path.length; i += 1) {
      const [x1, y1] = this.project(path[i - 1].x, path[i - 1].y);
      const [x2, y2] = this.project(path[i].x, path[i].y);
      ctx.strokeStyle = `rgba(200,169,97,${0.18 + 0.5 * (i / path.length)})`;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  },

  drawChampions(ctx, size) {
    const stamp = state.stamps[state.index];
    const frames = state.match.frames.filter((f) => f.ts_ms === stamp && f.x !== null);
    const radius = Math.max(13, size / 29);

    for (const frame of frames) {
      const focused = state.focusId === frame.participant_id;
      const dim = state.focusId && !focused;
      const [px, py] = this.project(frame.x || 0, frame.y || 0);
      const r = focused ? radius * 1.16 : radius;

      ctx.save();
      ctx.globalAlpha = dim ? 0.42 : 1;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.closePath();
      ctx.fillStyle = '#050d16';
      ctx.fill();
      ctx.clip();
      const image = this.icon(frame.champion_key || frame.champion);
      if (image && image.complete && image.naturalWidth) {
        ctx.drawImage(image, px - r, py - r, r * 2, r * 2);
      } else {
        ctx.fillStyle = frame.team_id === 100 ? '#12364f' : '#4a1c27';
        ctx.fillRect(px - r, py - r, r * 2, r * 2);
        ctx.fillStyle = '#f0e8d8';
        ctx.font = `600 ${Math.round(r * 0.78)}px system-ui`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText((frame.champion || '?').slice(0, 2), px, py);
      }
      ctx.restore();

      ctx.globalAlpha = dim ? 0.5 : 1;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.lineWidth = focused ? Math.max(3, size / 190) : Math.max(2, size / 240);
      ctx.strokeStyle = focused ? GOLD : (TEAM[frame.team_id] || '#6b7d8f');
      if (focused) { ctx.shadowColor = GOLD; ctx.shadowBlur = 12; }
      ctx.stroke();
      ctx.shadowBlur = 0;

      const hp = frame.health_max ? frame.health / frame.health_max : 1;
      if (hp < 0.999) {
        ctx.beginPath();
        ctx.arc(px, py, r + ctx.lineWidth + 2, -Math.PI / 2,
                -Math.PI / 2 + Math.PI * 2 * Math.max(0, Math.min(1, hp)));
        ctx.lineWidth = Math.max(2, size / 280);
        ctx.strokeStyle = hp > 0.5 ? '#4fb391' : hp > 0.25 ? '#d9a441' : '#e05c72';
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  },
};

/* -------------------------------------------------------------- GoldChart */

/* Team gold differential across the game.
 *
 * The single most useful macro view: it shows when a lead was built and when
 * it was thrown, and the markers say what caused each swing. Gold is summed
 * per team per frame, so it needs no extra API. */
const GoldChart = {
  series: [],

  compute() {
    const byStamp = new Map();
    for (const frame of state.match.frames) {
      if (!byStamp.has(frame.ts_ms)) byStamp.set(frame.ts_ms, { 100: 0, 200: 0 });
      const bucket = byStamp.get(frame.ts_ms);
      if (frame.team_id === 100 || frame.team_id === 200) {
        bucket[frame.team_id] += frame.total_gold || 0;
      }
    }
    this.series = [...byStamp.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([ts, teams]) => ({ ts, diff: teams[100] - teams[200] }));
  },

  draw(canvas) {
    if (!canvas || !state.match) return;
    if (!this.series.length) this.compute();
    const { ctx, width, height } = crisp(canvas, 148);
    if (this.series.length < 2) {
      ctx.fillStyle = '#6b7d8f';
      ctx.font = '12px system-ui';
      ctx.fillText('Not enough frames to chart gold.', 8, height / 2);
      return;
    }

    const pad = { l: 46, r: 8, t: 10, b: 16 };
    const plotW = width - pad.l - pad.r;
    const plotH = height - pad.t - pad.b;
    const peak = Math.max(2000, ...this.series.map((p) => Math.abs(p.diff)));
    const lastTs = this.series[this.series.length - 1].ts || 1;
    const X = (ts) => pad.l + (ts / lastTs) * plotW;
    const Y = (diff) => pad.t + plotH / 2 - (diff / peak) * (plotH / 2);

    // Axis labels and the zero line.
    ctx.font = '10px system-ui';
    ctx.fillStyle = '#6b7d8f';
    ctx.fillText(`+${Math.round(peak / 1000)}k`, 6, pad.t + 8);
    ctx.fillText('0', 6, Y(0) + 3);
    ctx.fillText(`-${Math.round(peak / 1000)}k`, 6, pad.t + plotH);
    ctx.strokeStyle = 'rgba(200,169,97,.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(0));
    ctx.lineTo(width - pad.r, Y(0));
    ctx.stroke();

    // Filled area, coloured by which side is ahead.
    for (const [sign, colour] of [[1, 'rgba(75,156,232,.22)'], [-1, 'rgba(224,92,114,.22)']]) {
      ctx.beginPath();
      ctx.moveTo(X(this.series[0].ts), Y(0));
      for (const point of this.series) {
        const value = sign > 0 ? Math.max(0, point.diff) : Math.min(0, point.diff);
        ctx.lineTo(X(point.ts), Y(value));
      }
      ctx.lineTo(X(this.series[this.series.length - 1].ts), Y(0));
      ctx.closePath();
      ctx.fillStyle = colour;
      ctx.fill();
    }

    // The line itself.
    ctx.beginPath();
    this.series.forEach((point, i) => {
      const x = X(point.ts);
      const y = Y(point.diff);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = GOLD;
    ctx.lineWidth = 1.6;
    ctx.stroke();

    // Event markers: what actually caused the swings.
    for (const moment of state.match.moments) {
      if (moment.kind !== 'objective' && moment.kind !== 'death') continue;
      if (moment.kind === 'death' && state.focusId
          && moment.details.victim_id !== state.focusId) continue;
      const x = X(moment.ts_ms);
      ctx.fillStyle = KIND_COLOUR[moment.kind];
      ctx.globalAlpha = moment.kind === 'objective' ? 0.95 : 0.5;
      if (moment.kind === 'objective') {
        ctx.beginPath();
        ctx.moveTo(x, pad.t - 2);
        ctx.lineTo(x + 3.5, pad.t + 4);
        ctx.lineTo(x, pad.t + 10);
        ctx.lineTo(x - 3.5, pad.t + 4);
        ctx.fill();
      } else {
        ctx.fillRect(x - 0.75, pad.t + plotH - 7, 1.5, 7);
      }
      ctx.globalAlpha = 1;
    }

    // Playhead.
    const now = state.stamps[state.index];
    if (now !== undefined) {
      ctx.strokeStyle = 'rgba(232,207,148,.85)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(X(now), pad.t);
      ctx.lineTo(X(now), pad.t + plotH);
      ctx.stroke();
    }
  },

  diffAt(ts) {
    if (!this.series.length) return 0;
    let best = this.series[0];
    for (const point of this.series) {
      if (Math.abs(point.ts - ts) < Math.abs(best.ts - ts)) best = point;
    }
    return best.diff;
  },
};

/* ------------------------------------------------------------- WaveRibbon */

/* Per-lane wave position through the game, from your team's perspective.
 *
 * This is the view no other League tool has, because no other tool infers
 * wave state. Above the centre line the wave is pushing toward the enemy;
 * below it, toward you. Opacity encodes the estimator's confidence, so a
 * weak read is visibly weak rather than silently equal to a strong one. */
const WaveRibbon = {
  draw(canvas) {
    if (!canvas || !state.match) return;
    const { ctx, width, height } = crisp(canvas, 116);
    const lanes = ['TOP', 'MIDDLE', 'BOTTOM'];
    const team = state.focusTeam || 100;
    const rows = state.match.wave_states.filter((w) => w.team_id === team);

    if (!rows.length) {
      ctx.fillStyle = '#6b7d8f';
      ctx.font = '12px system-ui';
      ctx.fillText('No wave states were derived for this game.', 8, height / 2);
      return;
    }

    const lastTs = state.stamps[state.stamps.length - 1] || 1;
    const laneH = height / lanes.length;

    lanes.forEach((lane, row) => {
      const top = row * laneH;
      const mid = top + laneH / 2;

      ctx.strokeStyle = 'rgba(168,182,196,.13)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(46, mid);
      ctx.lineTo(width - 6, mid);
      ctx.stroke();

      ctx.fillStyle = '#6b7d8f';
      ctx.font = '10px system-ui';
      ctx.fillText(lane, 6, mid + 3);

      const points = rows.filter((w) => w.lane === lane).sort((a, b) => a.ts_ms - b.ts_ms);
      for (const point of points) {
        const x = 46 + (point.ts_ms / lastTs) * (width - 52);
        const offset = Math.max(-1, Math.min(1, point.position || 0));
        const y = mid - offset * (laneH / 2 - 6);
        const confidence = Math.max(0.15, Math.min(1, point.confidence || 0.4));
        ctx.globalAlpha = confidence;
        ctx.fillStyle = offset > 0.1 ? '#4b9ce8' : offset < -0.1 ? '#e05c72' : '#6b7d8f';
        ctx.fillRect(x - 1, Math.min(mid, y), 2, Math.max(1.5, Math.abs(mid - y)));
        ctx.globalAlpha = 1;
      }
    });

    const now = state.stamps[state.index];
    if (now !== undefined) {
      ctx.strokeStyle = 'rgba(232,207,148,.8)';
      ctx.beginPath();
      ctx.moveTo(46 + (now / lastTs) * (width - 52), 0);
      ctx.lineTo(46 + (now / lastTs) * (width - 52), height);
      ctx.stroke();
    }
  },
};

/* ------------------------------------------------------------- top region */

async function loadStatus() {
  try {
    state.status = await api.get('/api/status');
  } catch (error) {
    toast(`Could not read local status: ${error.message}`, true);
    return;
  }
  const pill = $('#riotPill');
  const configured = state.status.riot_configured;
  pill.className = `pill ${configured ? 'ok' : 'warn'}`;
  pill.lastElementChild.textContent = configured ? 'Riot key active' : 'No Riot key';
  renderSetup();
}

function renderSetup() {
  const box = $('#setup');
  const status = state.status;
  if (!status) return;

  if (!status.riot_configured) {
    box.innerHTML = `
      <div class="card setup">
        <div class="eyebrow">First-time setup</div>
        <h2>Three steps to your first review</h2>
        <div class="steps">
          <div class="step active"><span class="n">I</span>Add your Riot key</div>
          <div class="step"><span class="n">II</span>Collect your games</div>
          <div class="step"><span class="n">III</span>Review your moments</div>
        </div>
        <p class="muted">Create a key at the Riot Developer Portal. It is validated with Riot,
          stored only on this PC, and never displayed again.</p>
        <form id="onboardForm" class="form-row">
          <input id="riotKey" class="input" type="password" autocomplete="off"
                 placeholder="Paste your Riot API key" required>
          <input id="riotId" class="input" autocomplete="off" placeholder="Optional: Name#TAG">
          <button class="btn primary" type="submit">Validate &amp; save</button>
        </form>
        <div id="setupMsg" class="muted" style="margin-top:12px"></div>
      </div>`;
    $('#onboardForm').addEventListener('submit', submitOnboard);
    return;
  }

  // A key alone is not enough: without a linked Riot ID there is no way to
  // tell which games are yours, so a "sync my games" button would be a lie.
  if (!status.player_linked) {
    box.innerHTML = `
      <div class="card setup">
        <div class="eyebrow">Step II of III</div>
        <h2>Link your account</h2>
        <p class="muted">Your key is saved. Add your Riot ID so LoLCoach knows which games
          are yours. It is the name and tag shown in the client, like <strong>Faker#KR1</strong>.</p>
        <form id="linkForm" class="form-row">
          <input id="riotId" class="input" autocomplete="off" placeholder="GameName#TAG" required>
          <button class="btn primary" type="submit">Link account</button>
        </form>
        <div id="setupMsg" class="muted" style="margin-top:12px"></div>
      </div>`;
    $('#linkForm').addEventListener('submit', submitLink);
    return;
  }

  const mine = status.my_matches || 0;
  box.innerHTML = `
    <div class="card${mine ? '' : ' setup'}">
      <div class="card-head">
        <div>
          <div class="eyebrow">${mine ? 'The archive' : 'Step III of III'}</div>
          <h2>${mine ? `${mine} of your games ready to review` : 'Sync your games'}</h2>
          <p class="muted" style="margin-top:5px">
            ${mine ? `${(status.timeline_frames || 0).toLocaleString()} timeline frames · ${(status.wave_states || 0).toLocaleString()} wave states · region ${status.region || '?'}`
                   : `LoLCoach will download your recent games from ${status.region || 'your region'} and work out the moments worth reviewing.`}
          </p>
        </div>
        <button id="collectBtn" class="btn primary">${mine ? 'Sync latest' : 'Sync my games'}</button>
      </div>
      <div id="setupMsg" class="muted"></div>
    </div>`;
  $('#collectBtn').addEventListener('click', startCollection);
  pollCollection();
}

/* System requirements, checked against this machine.
 *
 * A static list tells someone what to buy. This tells them whether the app
 * will actually work here, which is the question they are really asking. */
async function renderRequirements() {
  const box = $('#setup');
  box.innerHTML = '<div class="card setup"><div class="skeleton" style="height:180px"></div></div>';
  let payload;
  try {
    payload = await api.get('/api/requirements');
  } catch (error) {
    box.innerHTML = `<div class="card setup"><p>${error.message}</p></div>`;
    return;
  }

  const icon = { ok: '✓', warn: '!', fail: '✕' };
  const colour = { ok: 'var(--jade)', warn: 'var(--amber)', fail: 'var(--garnet)' };
  const checks = payload.checks.map((c) => `
    <div class="player" style="grid-template-columns:22px 1fr auto;cursor:default">
      <span style="color:${colour[c.status] || 'var(--ink-3)'};font-weight:700">${icon[c.status] || '·'}</span>
      <span>${c.name}<br><span class="dim" style="font-size:11px">${c.detail}</span></span>
      <span class="dim" style="font-size:10.5px;text-transform:uppercase">${c.status}</span>
    </div>`).join('');

  const spec = payload.spec.map((group) => `
    <div style="margin-top:16px">
      <div class="team-title" style="color:var(--gold)">${group.tier}</div>
      <p class="dim" style="font-size:11.5px;margin-bottom:8px">${group.note}</p>
      ${group.items.map(([k, v]) => `
        <div class="player" style="grid-template-columns:150px 1fr;cursor:default">
          <span class="dim">${k}</span><span>${v}</span>
        </div>`).join('')}
    </div>`).join('');

  box.innerHTML = `
    <div class="card setup">
      <div class="card-head">
        <div><div class="eyebrow">System check</div><h2>Can this PC run LoLCoach?</h2></div>
        <button id="reqClose" class="btn ghost">Done</button>
      </div>
      <div style="margin-bottom:6px">${checks}</div>
      ${spec}
    </div>`;
  $('#reqClose').addEventListener('click', renderSetup);
}

/* Settings: change the Riot ID or replace the key.
 *
 * The previous version faked ``riot_configured: false`` to reopen the key
 * form, which meant a linked user had no route back to the Riot ID field at
 * all — the one thing they are most likely to have typed wrong. Both are
 * offered here, independently. */
function renderSettings() {
  const status = state.status || {};
  $('#setup').innerHTML = `
    <div class="card setup">
      <div class="card-head">
        <div>
          <div class="eyebrow">Settings</div>
          <h2>Account &amp; key</h2>
        </div>
        <button id="settingsClose" class="btn ghost">Done</button>
      </div>
      <p class="muted" style="margin-bottom:6px">
        ${status.player_linked
          ? `Linked${status.riot_id ? ` as <strong>${status.riot_id}</strong>` : ''}, region ${status.region || 'unknown'}.`
          : 'No account is linked yet.'}
      </p>
      <form id="linkForm" class="form-row">
        <input id="riotId" class="input" autocomplete="off"
               placeholder="GameName#TAG" value="${status.riot_id || ''}" required>
        <button class="btn primary" type="submit">${status.player_linked ? 'Change account' : 'Link account'}</button>
      </form>
      <p class="dim" style="margin-top:6px;font-size:11.5px">
        Use the name and tag exactly as the client shows them. The tag is the part
        after the #, and it is not always your region.
      </p>
      <form id="keyForm" class="form-row" style="margin-top:18px">
        <input id="riotKey" class="input" type="password" autocomplete="off"
               placeholder="Replace Riot API key (optional)">
        <button class="btn" type="submit">Save key</button>
      </form>
      <div id="setupMsg" class="muted" style="margin-top:12px"></div>
      <div id="personality"></div>
    </div>`;
  $('#linkForm').addEventListener('submit', submitLink);
  $('#keyForm').addEventListener('submit', submitOnboard);
  $('#settingsClose').addEventListener('click', renderSetup);
  renderPersonality($('#personality'));
}

async function submitLink(event) {
  event.preventDefault();
  const button = event.target.querySelector('button');
  const message = $('#setupMsg');
  const label = button.textContent;
  button.disabled = true;
  button.textContent = 'Asking Riot…';
  message.textContent = 'Looking up your account and finding which region serves it…';
  try {
    const result = await api.post('/api/onboard', { riot_id: $('#riotId').value }, 45_000);
    message.textContent = result.message;
    await loadStatus();
  } catch (error) {
    message.textContent = error.message;
  } finally {
    // Always restore the form. A failed lookup that leaves the button dead
    // traps the user with no way to correct a typo.
    button.disabled = false;
    button.textContent = label;
  }
}

async function submitOnboard(event) {
  event.preventDefault();
  const button = event.target.querySelector('button');
  const message = $('#setupMsg');
  const label = button.textContent;
  button.disabled = true;
  button.textContent = 'Checking with Riot…';
  message.textContent = 'Validating your key…';
  // On the first-run panel the Riot ID sits in the same form. In Settings it
  // has its own form, so the key form must not silently re-link the account.
  const body = { riot_key: $('#riotKey').value };
  const idField = $('#riotId');
  if (idField && idField.value.trim() && !$('#keyForm')) {
    body.riot_id = idField.value;
  }
  try {
    const result = await api.post('/api/onboard', body, 45_000);
    message.textContent = result.message;
    await loadStatus();
  } catch (error) {
    message.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

async function startCollection() {
  const button = $('#collectBtn');
  const message = $('#setupMsg');
  button.disabled = true;
  try {
    const result = await api.post('/api/collection', { limit: 25 });
    message.innerHTML = `${result.detail}<div class="progress"><i></i></div>`;
  } catch (error) {
    message.textContent = error.message;
    button.disabled = false;
    return;
  }
  pollCollection();
}

let collectionTimer = null;

/* Watch the sync worker.
 *
 * Deliberately idempotent and self-owning. The previous version cleared its
 * own timer on every call, and renderSetup() called it — while loadStatus()
 * calls renderSetup(). So a routine status refresh during a sync silently
 * stopped the progress updates while the work carried on in the background,
 * which looks exactly like the app freezing mid-sync. The timer is now only
 * cleared when the server reports the run is actually over. */
async function pollCollection() {
  if (collectionTimer) return;              // already watching
  const tick = async () => {
    let snapshot;
    try {
      snapshot = await api.get('/api/collection');
    } catch {
      return;                               // transient; keep watching
    }
    const running = snapshot.state === 'running';
    const message = $('#setupMsg');
    if (message && snapshot.state !== 'idle') {
      message.innerHTML = snapshot.detail + (running ? '<div class="progress"><i></i></div>' : '');
    }
    const button = $('#collectBtn');
    if (button) button.disabled = running;
    if (!running) {
      clearInterval(collectionTimer);
      collectionTimer = null;
      if (snapshot.state === 'failed') toast(snapshot.detail, true);
      if (snapshot.state === 'complete') {
        await loadStatus();
        await loadMatches();
      }
    }
  };
  await tick();
  if (!collectionTimer) collectionTimer = setInterval(tick, 2000);
}

/* ------------------------------------------------------------- match list */

async function loadMatches() {
  let matches;
  try {
    matches = await api.get(`/api/matches?limit=50&mine=${state.onlyMine ? 1 : 0}`);
  } catch (error) {
    $('#matchList').innerHTML = `<div class="empty"><p>${error.message}</p></div>`;
    return;
  }
  state.matches = matches;
  renderMatchList();
  if (matches.length && !state.match) selectMatch(matches[0].match_id);
}

function renderMatchList() {
  const list = $('#matchList');
  const query = state.filter.toLowerCase();
  const visible = state.matches.filter((m) =>
    `${m.match_id} ${m.patch || ''} ${m.champion || ''}`.toLowerCase().includes(query));

  $('#matchCount').textContent = state.matches.length || '';

  if (!state.matches.length) {
    list.innerHTML = `
      <div class="empty"><div class="big">◈</div><h3>No games yet</h3>
      <p>Add your Riot key, then collect your games.</p></div>`;
    return;
  }
  if (!visible.length) {
    list.innerHTML = '<div class="empty"><p>Nothing matches that filter.</p></div>';
    return;
  }

  list.innerHTML = '';
  for (const match of visible) {
    const won = match.win;
    const chip = won === 1 ? '<span class="result-chip win">VICTORY</span>'
      : won === 0 ? '<span class="result-chip loss">DEFEAT</span>'
      : '<span class="result-chip none">—</span>';
    const duration = match.duration_s ? `${Math.round(match.duration_s / 60)} min` : 'unknown length';
    const button = el('button', 'match-item');
    button.setAttribute('role', 'listitem');
    button.innerHTML = `
      <div class="row">${chip}<span class="id">${match.champion || match.match_id}</span></div>
      <div class="sub">${duration} · patch ${match.patch || '?'} · ${match.moments || 0} moments</div>`;
    button.addEventListener('click', () => selectMatch(match.match_id));
    if (state.match && state.match.match.match_id === match.match_id) {
      button.setAttribute('aria-current', 'true');
    }
    list.append(button);
  }
}

/* ------------------------------------------------------------ match detail */

async function selectMatch(matchId) {
  stopPlayback();
  $('#review').innerHTML = '<div class="card"><div class="skeleton" style="height:340px"></div></div>';
  let detail;
  try {
    detail = await api.get(`/api/matches/${encodeURIComponent(matchId)}`);
  } catch (error) {
    $('#review').innerHTML = `<div class="empty"><h3>Could not open that game</h3><p>${error.message}</p></div>`;
    return;
  }
  state.match = detail;
  state.stamps = [...new Set(detail.frames.map((f) => f.ts_ms))].sort((a, b) => a - b);
  state.index = 0;
  state.activeMomentId = null;
  // Centre the review on the viewer when we can identify them: their deaths,
  // their trail, their heatmap. A personal review that opens on a stranger's
  // mistakes is not a personal review.
  state.focusId = detail.you ?? null;
  const mePlayer = detail.participants.find((p) => p.participant_id === state.focusId);
  state.focusTeam = mePlayer ? mePlayer.team_id : 100;
  GoldChart.series = [];
  renderMatchList();
  renderReview();
}

function setFocus(participantId) {
  state.focusId = state.focusId === participantId ? null : participantId;
  const player = state.match.participants.find((p) => p.participant_id === state.focusId);
  state.focusTeam = player ? player.team_id : 100;
  renderScoreboard();
  redrawAll();
}

function redrawAll() {
  MapView.draw();
  GoldChart.draw($('#gold'));
  WaveRibbon.draw($('#ribbon'));
  drawTicks();
}

function renderReview() {
  const detail = state.match;
  const match = detail.match;
  const counts = (kind) => detail.moments.filter((m) => m.kind === kind).length;

  $('#review').innerHTML = `
    <div class="card">
      <div class="card-head">
        <div>
          <div class="eyebrow">Match review</div>
          <h2>${match.match_id}</h2>
          <p class="muted" style="margin-top:5px">
            ${match.source} · patch ${match.patch || 'unknown'} ·
            ${match.duration_s ? Math.round(match.duration_s / 60) + ' minutes' : 'unknown length'} ·
            ${detail.frames.length.toLocaleString()} frames
          </p>
        </div>
      </div>
      <div class="stats">
        <div class="stat"><b>${detail.moments.length}</b><span>moments</span></div>
        <div class="stat"><b>${counts('death')}</b><span>deaths</span></div>
        <div class="stat"><b>${counts('wave')}</b><span>wave calls</span></div>
        <div class="stat"><b>${counts('objective')}</b><span>objectives</span></div>
        <div class="stat"><b>${counts('recall')}</b><span>recalls</span></div>
      </div>
    </div>

    <div class="split">
      <div>
        <div class="card">
          <div class="card-head">
            <h3>Field</h3>
            <div class="map-tools">
              <button id="trailBtn" class="chip" aria-pressed="${state.showTrails}">Trail</button>
              <button id="deathBtn" class="chip" aria-pressed="${state.showDeaths}">Deaths</button>
            </div>
          </div>
          <div class="map-wrap"><canvas id="map" class="map-canvas" aria-label="Match position map"></canvas></div>
          <div class="map-legend">
            <span class="key"><i class="dot blue"></i>Blue</span>
            <span class="key"><i class="dot red"></i>Red</span>
            <span class="key">outer ring = health</span>
            <span class="key">gold ring = focused player</span>
          </div>
          <div class="transport">
            <button id="playBtn" class="btn icon" title="Play/pause (Space)" aria-label="Play">▶</button>
            <span id="clock" class="clock">0:00</span>
            <div class="scrub">
              <canvas id="ticks" class="ticks" aria-hidden="true"></canvas>
              <input id="slider" type="range" min="0" max="${Math.max(0, state.stamps.length - 1)}" value="0"
                     aria-label="Timeline position">
            </div>
          </div>
          <p class="dim" style="margin-top:10px;font-size:11px">
            <kbd>Space</kbd> play · <kbd>←</kbd><kbd>→</kbd> step · <kbd>J</kbd><kbd>K</kbd> moments · <kbd>D</kbd> deaths
          </p>
        </div>

        <div class="card">
          <div class="card-head"><h3>Gold differential</h3><span id="goldNow" class="numeral" style="color:var(--gold-bright)"></span></div>
          <canvas id="gold" class="chart" aria-label="Team gold differential over time"></canvas>
          <p class="chart-note">Above the line, blue side leads. Diamonds mark objectives; ticks mark deaths.</p>
        </div>

        <div class="card">
          <div class="card-head"><h3>Wave pressure</h3></div>
          <canvas id="ribbon" class="ribbon" aria-label="Wave position per lane over time"></canvas>
          <p class="chart-note">Above each lane line the wave pushes toward the enemy; below, toward you.
            Fainter marks are lower-confidence estimates.</p>
        </div>

        <div class="card">
          <div class="card-head">
            <h3>Your coach</h3>
            <span id="coachPill" class="pill"><i class="dot"></i><span>checking…</span></span>
          </div>
          <div id="insights"></div>
          <div id="chatLog" class="chat-log"></div>
          <div id="starters" class="chips"></div>
          <form id="askForm" class="form-row" style="margin-top:0">
            <input id="question" class="input" placeholder="Ask anything about this game…"
                   aria-label="Question about the selected moment">
            <button class="btn primary" type="submit">Ask</button>
          </form>
          <p class="dim" style="margin-top:8px;font-size:11px">
            Runs entirely on this PC, so ask as much as you like. Answers use the
            evidence stored at the selected timestamp.
          </p>
        </div>
      </div>

      <div>
        <div class="card">
          <h3 style="margin-bottom:12px">Coaching moments</h3>
          <div class="chips" id="chips"></div>
          <div class="moments" id="moments"></div>
        </div>
        <div class="card">
          <div class="card-head"><h3>Scoreboard</h3><span class="dim" style="font-size:11px">click to focus</span></div>
          <div id="scoreboard" class="teams"></div>
        </div>
      </div>
    </div>`;

  MapView.attach($('#map'), detail.map_bounds);
  renderScoreboard();
  renderChips();
  renderMoments();
  // Minute zero has all ten champions stacked on two fountains, which reads
  // as an empty map. Open on the moment the analysis ranked highest instead.
  const opening = visibleMoments()[0];
  seek(opening ? nearestFrame(opening.ts_ms) : Math.floor(state.stamps.length * 0.25));
  loadInsights();
  pollCoach();

  $('#slider').addEventListener('input', (event) => seek(Number(event.target.value)));
  $('#playBtn').addEventListener('click', togglePlayback);
  $('#askForm').addEventListener('submit', askQuestion);
  $('#trailBtn').addEventListener('click', () => toggleLayer('showTrails', '#trailBtn'));
  $('#deathBtn').addEventListener('click', () => toggleLayer('showDeaths', '#deathBtn'));
}

function toggleLayer(key, selector) {
  state[key] = !state[key];
  const button = $(selector);
  if (button) button.setAttribute('aria-pressed', String(state[key]));
  MapView.draw();
}

function renderScoreboard() {
  const box = $('#scoreboard');
  if (!box) return;
  box.innerHTML = [100, 200].map((team) => `
    <div>
      <div class="team-title ${team === 100 ? 'blue' : 'red'}">${team === 100 ? 'Blue' : 'Red'} side</div>
      ${state.match.participants.filter((p) => p.team_id === team).map((p) => `
        <div class="player" data-pid="${p.participant_id}" aria-current="${state.focusId === p.participant_id}">
          ${p.champion_key
            ? `<img src="/assets/champion/${encodeURIComponent(p.champion_key)}" alt="" loading="lazy">`
            : `<span class="fallback">${(p.champion || '?').slice(0, 2)}</span>`}
          <span>${p.champion || 'Unknown'}<br><span class="dim" style="font-size:10.5px">${p.role || ''}</span></span>
          <span class="kda tabular">${p.kills}/${p.deaths}/${p.assists}</span>
        </div>`).join('')}
    </div>`).join('');
  box.querySelectorAll('.player').forEach((node) => {
    node.addEventListener('click', () => setFocus(Number(node.dataset.pid)));
  });
}

/* -------------------------------------------------------------- transport */

function seek(index) {
  state.index = Math.max(0, Math.min(state.stamps.length - 1, index));
  const slider = $('#slider');
  if (slider) slider.value = String(state.index);
  const now = state.stamps[state.index] || 0;
  const clockNode = $('#clock');
  if (clockNode) clockNode.textContent = clock(now);
  const goldNode = $('#goldNow');
  if (goldNode) {
    if (!GoldChart.series.length) GoldChart.compute();
    const diff = GoldChart.diffAt(now);
    const side = diff >= 0 ? 'blue' : 'red';
    goldNode.textContent = `${diff >= 0 ? '+' : ''}${diff.toLocaleString()} ${side}`;
    goldNode.style.color = diff >= 0 ? TEAM[100] : TEAM[200];
  }
  redrawAll();
}

function nearestFrame(ts) {
  let best = 0;
  state.stamps.forEach((stamp, i) => {
    if (Math.abs(stamp - ts) < Math.abs(state.stamps[best] - ts)) best = i;
  });
  return best;
}

function step(delta) { seek(state.index + delta); }

function togglePlayback() { state.playing ? stopPlayback() : startPlayback(); }

function startPlayback() {
  if (!state.stamps.length) return;
  state.playing = true;
  const button = $('#playBtn');
  if (button) { button.textContent = '❚❚'; button.setAttribute('aria-label', 'Pause'); }
  state.playTimer = setInterval(() => {
    if (state.index >= state.stamps.length - 1) { stopPlayback(); return; }
    step(1);
  }, 650);
}

function stopPlayback() {
  state.playing = false;
  clearInterval(state.playTimer);
  const button = $('#playBtn');
  if (button) { button.textContent = '▶'; button.setAttribute('aria-label', 'Play'); }
}

/* Event ticks under the scrubber, so the action is findable without
 * scrubbing blindly through half an hour of game. */
function drawTicks() {
  const canvas = $('#ticks');
  if (!canvas || !state.match) return;
  const { ctx, width, height } = crisp(canvas, 22);
  const last = state.stamps[state.stamps.length - 1] || 1;
  for (const moment of state.match.moments) {
    if (state.focusId && moment.kind === 'death'
        && moment.details.victim_id !== state.focusId) continue;
    const x = (moment.ts_ms / last) * width;
    ctx.fillStyle = KIND_COLOUR[moment.kind] || '#6b7d8f';
    ctx.globalAlpha = moment.kind === 'rotation' ? 0.28 : 0.85;
    ctx.fillRect(x - 0.5, height * 0.28, 1.4, height * 0.44);
  }
  ctx.globalAlpha = 1;
}

/* ---------------------------------------------------------------- moments */

function renderChips() {
  const kinds = ['all', 'death', 'wave', 'recall', 'objective', 'rotation'];
  const box = $('#chips');
  box.innerHTML = '';
  for (const kind of kinds) {
    const count = kind === 'all'
      ? state.match.moments.length
      : state.match.moments.filter((m) => m.kind === kind).length;
    if (!count && kind !== 'all') continue;
    const chip = el('button', 'chip', `${kind === 'all' ? 'All' : kind} ${count}`);
    chip.setAttribute('aria-pressed', String(state.momentKind === kind));
    chip.addEventListener('click', () => { state.momentKind = kind; renderChips(); renderMoments(); });
    box.append(chip);
  }
}

function visibleMoments() {
  return state.match.moments
    .filter((m) => state.momentKind === 'all' || m.kind === state.momentKind)
    .filter((m) => !state.focusId || m.kind !== 'death'
      || m.details.victim_id === state.focusId)
    .slice(0, 40);
}

function renderMoments() {
  const box = $('#moments');
  const items = visibleMoments();
  if (!items.length) {
    box.innerHTML = '<div class="empty"><h3>Nothing here</h3><p>No moments in this category.</p></div>';
    return;
  }
  box.innerHTML = '';
  items.forEach((moment, position) => {
    const id = `${moment.kind}-${moment.ts_ms}-${position}`;
    const node = el('button', `moment ${moment.kind}`);
    const confidence = moment.confidence === null || moment.confidence === undefined ? '' : `
      <div class="conf">
        <span>confidence</span>
        <span class="conf-bar ${moment.confidence >= 0.7 ? '' : moment.confidence >= 0.5 ? 'mid' : 'low'}">
          <i style="width:${Math.round(moment.confidence * 100)}%"></i>
        </span>
        <span class="tabular">${Number(moment.confidence).toFixed(2)}</span>
      </div>`;
    node.innerHTML = `
      <div class="top"><span class="t">${clock(moment.ts_ms)}</span><span class="title">${moment.title}</span></div>
      <div class="body">${moment.summary}</div>${confidence}`;
    node.setAttribute('aria-current', String(state.activeMomentId === id));
    node.addEventListener('click', () => jumpToMoment(moment, id));
    box.append(node);
  });
}

function jumpToMoment(moment, id) {
  state.activeMomentId = id;
  stopPlayback();
  seek(nearestFrame(moment.ts_ms));
  renderMoments();
  $('#map').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function cycleMoment(delta) {
  const items = visibleMoments();
  if (!items.length) return;
  const current = items.findIndex((m, position) => `${m.kind}-${m.ts_ms}-${position}` === state.activeMomentId);
  const next = Math.max(0, Math.min(items.length - 1, (current < 0 ? 0 : current) + delta));
  jumpToMoment(items[next], `${items[next].kind}-${items[next].ts_ms}-${next}`);
}

/* --------------------------------------------------------------- insights */

/* Instant, model-free takeaways shown the moment a game opens.
 *
 * The point is that the coach has something to say before you ask. These are
 * computed from stored features, so they appear immediately; the model is only
 * spent when you click one and want the reasoning. */
async function loadInsights() {
  const box = $('#insights');
  if (!box || !state.match) return;
  let payload;
  try {
    payload = await api.get(
      `/api/matches/${encodeURIComponent(state.match.match.match_id)}/insights`);
  } catch {
    box.innerHTML = '';
    return;
  }

  const cards = (payload.highlights || []).map((item, index) => `
    <button class="moment ${item.kind === 'pattern' ? 'death' : item.kind}" data-i="${index}">
      <div class="top"><span class="t">${clock(item.ts_ms)}</span><span class="title">${item.headline}</span></div>
      <div class="body">${item.detail || ''}</div>
    </button>`).join('');

  box.innerHTML = cards
    ? `<p class="dim" style="font-size:11px;margin-bottom:8px">
         ${payload.player_identified ? 'What stood out in your game' : 'What stood out in this game'}
         — click one to ask about it.
       </p>
       <div class="moments" style="max-height:none;margin-bottom:14px">${cards}</div>`
    : '';

  box.querySelectorAll('.moment').forEach((node) => {
    const item = payload.highlights[Number(node.dataset.i)];
    node.addEventListener('click', () => {
      const stamp = state.stamps.reduce((best, ts, i) =>
        Math.abs(ts - item.ts_ms) < Math.abs(state.stamps[best] - item.ts_ms) ? i : best, 0);
      seek(stamp);
      askDirect(item.question);
    });
  });

  const starters = $('#starters');
  if (starters) {
    starters.innerHTML = '';
    for (const text of payload.starters || []) {
      const chip = el('button', 'chip', text);
      chip.addEventListener('click', () => askDirect(text));
      starters.append(chip);
    }
  }
}

/* -------------------------------------------------------------- coach state */

let coachTimer = null;
async function pollCoach() {
  const paint = (snapshot) => {
    const pill = $('#coachPill');
    if (!pill) return;
    const map = {
      loading: ['warn', 'warming up…'],
      ready: ['ok', 'ready'],
      error: ['warn', 'unavailable'],
    };
    const [cls, label] = map[snapshot.state] || ['', snapshot.state];
    pill.className = `pill ${cls}`;
    pill.lastElementChild.textContent = label;
    pill.title = snapshot.detail || '';
  };
  const tick = async () => {
    let snapshot;
    try { snapshot = await api.get('/api/coach'); } catch { return; }
    paint(snapshot);
    if (snapshot.state !== 'loading') {
      clearInterval(coachTimer);
      coachTimer = null;
    }
  };
  await tick();
  if (!coachTimer) coachTimer = setInterval(tick, 3000);
}

/* ------------------------------------------------------------------- chat */

function askDirect(question) {
  const input = $('#question');
  if (!input) return;
  input.value = question;
  $('#askForm').requestSubmit();
  $('#chatLog').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

async function askQuestion(event) {
  event.preventDefault();
  const input = $('#question');
  const question = input.value.trim();
  if (!question) return;
  const log = $('#chatLog');
  log.append(el('div', 'bubble you', question));
  input.value = '';

  const answer = el('div', 'bubble coach caret', '');
  log.append(answer);
  log.scrollTop = log.scrollHeight;

  const stamp = state.stamps[state.index] || 0;
  try {
    const response = await fetch(
      `/api/matches/${encodeURIComponent(state.match.match.match_id)}/ask/stream`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ question, timestamp_ms: stamp,
                               tone: prefs.load().tone }),
      });
    if (!response.ok || !response.body) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || 'The local coach is unavailable.');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let text = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
      answer.textContent = text;
      log.scrollTop = log.scrollHeight;
    }
    answer.classList.remove('caret');
    if (!text.trim()) answer.textContent = 'The local coach returned an empty answer.';
  } catch (error) {
    answer.remove();
    log.append(el('div', 'bubble error', error.message));
    log.scrollTop = log.scrollHeight;
  }
}

/* -------------------------------------------------------------- bootstrap */

function onKeyDown(event) {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  const actions = {
    ' ': () => { event.preventDefault(); togglePlayback(); },
    arrowleft: () => step(-1),
    arrowright: () => step(1),
    j: () => cycleMoment(1),
    k: () => cycleMoment(-1),
    d: () => toggleLayer('showDeaths', '#deathBtn'),
    t: () => toggleLayer('showTrails', '#trailBtn'),
    r: () => refreshAll(),
  };
  const handler = actions[event.key] || actions[key];
  if (handler) handler();
}

async function refreshAll() {
  await loadStatus();
  await loadMatches();
}

function boot() {
  applyAccent(prefs.load().accent || 'brass');
  $('#search').addEventListener('input', (event) => {
    state.filter = event.target.value;
    renderMatchList();
  });
  $('#mineToggle').addEventListener('click', () => {
    state.onlyMine = !state.onlyMine;
    $('#mineToggle').setAttribute('aria-pressed', String(state.onlyMine));
    state.match = null;
    loadMatches();
  });
  $('#refreshBtn').addEventListener('click', refreshAll);
  $('#progressBtn').addEventListener('click', () => {
    renderProgress();
    $('#setup').scrollIntoView({ behavior: 'smooth' });
  });
  $('#reqBtn').addEventListener('click', () => {
    renderRequirements();
    $('#setup').scrollIntoView({ behavior: 'smooth' });
  });
  $('#settingsBtn').addEventListener('click', () => {
    if (!state.status) return;
    renderSettings();
    $('#setup').scrollIntoView({ behavior: 'smooth' });
  });
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('resize', () => {
    if (state.match) { MapView.attach($('#map'), state.match.map_bounds); redrawAll(); }
  });
  refreshAll();
}

document.addEventListener('DOMContentLoaded', boot);

/* ==========================================================================
 * Personality, theme and progress.
 *
 * Preferences live in localStorage rather than on the server: they are per
 * person, not per install, and nothing here needs to survive a reinstall
 * badly enough to justify another config file.
 * ======================================================================= */

const ACCENTS = {
  brass:   { label: 'Brass',   gold: '#c8a961', bright: '#e8cf94', dim: '#8a7440' },
  arcane:  { label: 'Arcane',  gold: '#58c8d8', bright: '#9ce8f2', dim: '#3d8a96' },
  crimson: { label: 'Crimson', gold: '#d9556b', bright: '#f0909f', dim: '#8f3644' },
  jade:    { label: 'Jade',    gold: '#5fbf95', bright: '#96e0bd', dim: '#3a7a60' },
};

const prefs = {
  load() {
    try {
      return JSON.parse(localStorage.getItem('lolcoach.prefs') || '{}');
    } catch { return {}; }
  },
  save(patch) {
    const next = { ...this.load(), ...patch };
    try { localStorage.setItem('lolcoach.prefs', JSON.stringify(next)); } catch { /* private mode */ }
    return next;
  },
};

function applyAccent(name) {
  const accent = ACCENTS[name] || ACCENTS.brass;
  const root = document.documentElement.style;
  root.setProperty('--gold', accent.gold);
  root.setProperty('--gold-bright', accent.bright);
  root.setProperty('--gold-dim', accent.dim);
  root.setProperty('--gold-line', hexToRgba(accent.gold, 0.28));
  root.setProperty('--gold-glow', hexToRgba(accent.gold, 0.12));
}

function hexToRgba(hex, alpha) {
  const value = hex.replace('#', '');
  const int = parseInt(value.length === 3
    ? value.split('').map((c) => c + c).join('') : value, 16);
  if (Number.isNaN(int)) return `rgba(200,169,97,${alpha})`;
  return `rgba(${(int >> 16) & 255}, ${(int >> 8) & 255}, ${int & 255}, ${alpha})`;
}

/* ------------------------------------------------------------- progress */

async function renderProgress() {
  const box = $('#setup');
  box.innerHTML = '<div class="card"><div class="skeleton" style="height:170px"></div></div>';
  let data;
  try {
    data = await api.get('/api/progress');
  } catch (error) {
    box.innerHTML = `<div class="card"><p>${error.message}</p></div>`;
    return;
  }

  if (!data.linked) {
    box.innerHTML = `<div class="card setup"><div class="eyebrow">Progress</div>
      <h2>Link your account first</h2>
      <p class="muted">Progress tracks your own habits across games, so it needs to
        know which player you are.</p>
      <button id="progLink" class="btn primary" style="margin-top:12px">Open settings</button></div>`;
    $('#progLink').addEventListener('click', renderSettings);
    return;
  }
  if (!data.games) {
    box.innerHTML = `<div class="card setup"><div class="eyebrow">Progress</div>
      <h2>No games of yours yet</h2>
      <p class="muted">Sync your games and this fills in automatically.</p></div>`;
    return;
  }

  const focus = data.focus;
  const trend = data.trend || {};
  const arrow = { better: '▼', worse: '▲', flat: '■' }[trend.direction] || '■';
  const trendWord = { better: 'improving', worse: 'slipping', flat: 'holding' }[trend.direction] || '';
  const peak = Math.max(...(data.causes || []).map((c) => c.count), 1);

  box.innerHTML = `
    <div class="card focus-card">
      <div class="focus-head">
        <div class="eyebrow">Work on this</div>
        <span class="trend ${trend.direction}">${arrow} ${trendWord} · ${trend.recent} deaths/game</span>
      </div>
      <h2 style="margin:6px 0">${focus ? focus.headline : 'No repeating pattern yet'}</h2>
      <p class="muted">${focus ? focus.detail : 'Play a few more games and a habit will show up here.'}</p>
      ${focus ? '<button id="focusAsk" class="btn primary" style="margin-top:14px">Ask the coach how to fix it</button>' : ''}
      <div class="bars">
        ${(data.causes || []).map((c) => `
          <div class="bar-row">
            <span class="dim">${c.cause}</span>
            <span class="bar"><i style="width:${Math.round((c.count / peak) * 100)}%"></i></span>
            <span class="tabular">${c.count}</span>
          </div>`).join('')}
      </div>
      <div class="stats" style="margin-top:16px">
        <div class="stat"><b>${data.games}</b><span>games tracked</span></div>
        <div class="stat"><b>${Math.round((data.winrate || 0) * 100)}%</b><span>win rate</span></div>
        <div class="stat"><b>${data.deaths_per_game}</b><span>deaths / game</span></div>
        <div class="stat"><b>${trend.older} → ${trend.recent}</b><span>then / now</span></div>
      </div>
      <button id="progClose" class="btn ghost" style="margin-top:14px">Done</button>
    </div>`;

  $('#progClose').addEventListener('click', renderSetup);
  const ask = $('#focusAsk');
  if (ask && focus) {
    ask.addEventListener('click', () => {
      renderSetup();
      if (state.match) askDirect(focus.question);
      else toast('Open one of your games first, then ask.');
    });
  }
}

/* ---------------------------------------------------------- personality */

async function renderPersonality(container) {
  let payload;
  try { payload = await api.get('/api/tones'); } catch { return; }
  const chosen = prefs.load().tone || payload.default;
  const accent = prefs.load().accent || 'brass';

  container.innerHTML = `
    <h3 style="margin-top:18px">Coach personality</h3>
    <p class="dim" style="font-size:11.5px;margin-top:4px">
      Changes how the coach talks. It is held to the same evidence either way.
    </p>
    <div class="opt-grid" id="toneGrid">
      ${payload.tones.map((t) => `
        <button class="opt" data-tone="${t.id}" aria-pressed="${t.id === chosen}">
          <strong>${t.label}</strong><span>${t.blurb}</span>
        </button>`).join('')}
    </div>
    <h3 style="margin-top:18px">Accent</h3>
    <div class="swatches" id="accentRow">
      ${Object.entries(ACCENTS).map(([key, a]) => `
        <button class="swatch" data-accent="${key}" title="${a.label}"
                aria-pressed="${key === accent}"
                style="background:linear-gradient(155deg, ${a.bright}, ${a.dim})"></button>`).join('')}
    </div>`;

  container.querySelectorAll('[data-tone]').forEach((node) => {
    node.addEventListener('click', () => {
      prefs.save({ tone: node.dataset.tone });
      container.querySelectorAll('[data-tone]').forEach((n) =>
        n.setAttribute('aria-pressed', String(n === node)));
      toast(`Coach set to ${node.querySelector('strong').textContent}.`);
    });
  });
  container.querySelectorAll('[data-accent]').forEach((node) => {
    node.addEventListener('click', () => {
      prefs.save({ accent: node.dataset.accent });
      applyAccent(node.dataset.accent);
      container.querySelectorAll('[data-accent]').forEach((n) =>
        n.setAttribute('aria-pressed', String(n === node)));
    });
  });
}
