/* EchoMoE Figure 4: what makes compression work (Qwen3-30B-A3B).
   (a) Table "Dispatch compression across code widths and reference strides" (stride 4 and plain
       dispatch, combine at full width) and Table "Combine decoder variants" (direct decoding,
       dispatch at full width); score changes are relative to the uncompressed model (66.92%).
   (b) Same dispatch table, all strides. RDMA saved uses the paper's traffic model (Eq. "Communication
       Cost"): d = 2048, 2.73 remote nodes per routed transfer, references sent to 3 remote nodes,
       combine at full width. This reproduces the paper's reported reductions (e.g. 2.39x, 4.03x).
   (c) Table "Throughput and latency across workloads", Decode TPOT speedup rows. */
(() => {
  'use strict';
  const root = document.querySelector('.fig4');
  if (!root) return;
  const NS = 'http://www.w3.org/2000/svg';
  // panel height for a panel drawn at its real pixel width
  const heightFor = W => Math.round(Math.max(250, Math.min(290, W * 0.8)));
  const el = (tag, attrs, parent, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  };
  const path = pts => pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('');
  const pp = v => (v < 0 ? '−' : v > 0 ? '+' : '') + Math.abs(v).toFixed(1) + ' pp';
  const BASE = 66.92;

  // ---------- tooltip ----------
  const tip = root.querySelector('.f4-tip');
  function showAt(target, html) {
    tip.innerHTML = html; tip.hidden = false;
    const box = root.getBoundingClientRect(), r = target.getBoundingClientRect();
    let x = r.left - box.left + r.width / 2 + 12;
    if (x + tip.offsetWidth > box.width - 6) x = r.left - box.left + r.width / 2 - tip.offsetWidth - 12;
    tip.style.left = Math.max(4, x) + 'px';
    tip.style.top = (r.top - box.top + r.height / 2) + 'px';
  }
  const hide = () => { tip.hidden = true; };
  function hit(svg, cx, cy, html) {
    const h = el('circle', { cx, cy, r: 10, class: 'f4-hit', tabindex: 0 }, svg);
    h.addEventListener('mouseenter', () => showAt(h, html));
    h.addEventListener('mouseleave', hide);
    h.addEventListener('focus', () => showAt(h, html));
    h.addEventListener('blur', hide);
  }
  function marker(svg, shape, cx, cy, cls, size = 4.5) {
    if (shape === 'circle') return el('circle', { cx, cy, r: size, class: cls }, svg);
    if (shape === 'square') return el('rect', { x: cx - size, y: cy - size, width: 2 * size, height: 2 * size, rx: 1.2, class: cls }, svg);
    if (shape === 'diamond') { const s = size * 1.3; return el('path', { d: `M${cx},${cy - s}L${cx + s},${cy}L${cx},${cy + s}L${cx - s},${cy}Z`, class: cls }, svg); }
    const s = size * 1.25; // triangle
    return el('path', { d: `M${cx},${cy - s}L${cx + s},${cy + s * 0.8}L${cx - s},${cy + s * 0.8}Z`, class: cls }, svg);
  }
  function frame(svg, W, H, M, yTicks, Y, yFmt, yLabel, xLabel) {
    for (const t of yTicks) {
      el('line', { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), class: 'f4-grid' }, svg);
      el('text', { x: M.l - 7, y: Y(t) + 4, class: 'f4-tick', 'text-anchor': 'end' }, svg, yFmt(t));
    }
    el('line', { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: 'f4-axis' }, svg);
    el('text', { x: (M.l + W - M.r) / 2, y: H - 6, class: 'f4-axis-label', 'text-anchor': 'middle' }, svg, xLabel);
    const my = (M.t + H - M.b) / 2;
    el('text', { x: 13, y: my, class: 'f4-axis-label', 'text-anchor': 'middle', transform: `rotate(-90 13 ${my})` }, svg, yLabel);
  }

  // ---------- (a) compression-path ablation ----------
  EchoFit.mount(root.querySelector('[data-f4="a"]'), W => {
    const svg = root.querySelector('[data-f4="a"]');
    const H = heightFor(W), M = { l: 46, r: 16, t: 14, b: 46 };
    const widths = [1280, 1024, 768, 512, 256];
    const X = w => M.l + 22 + widths.indexOf(w) * ((W - M.l - M.r - 44) / 4);
    const Y = v => M.t + (3 - v) / 33 * (H - M.t - M.b);
    frame(svg, W, H, M, [0, -10, -20, -30], Y, t => (t < 0 ? '−' : '') + Math.abs(t), 'Score change (pp) →', 'Code width');
    for (const w of widths) el('text', { x: X(w), y: H - M.b + 17, class: 'f4-tick', 'text-anchor': 'middle' }, svg, w);
    const REF = [[1280, -0.0], [1024, -0.6], [768, -1.0], [512, -2.6], [256, -5.6]];
    const PLAIN = [[1024, -7.1], [768, -12.4], [512, -21.1]];
    const COMB = [[1280, 65.2], [1024, 64.1], [768, 60.6], [512, 54.7], [256, 41.4]].map(([w, s]) => [w, +(s - BASE).toFixed(1), s]);
    el('path', { d: path(PLAIN.map(d => [X(d[0]), Y(d[1])])), class: 'f4-line f4-plain' }, svg);
    el('path', { d: path(COMB.map(d => [X(d[0]), Y(d[1])])), class: 'f4-line f4-comb' }, svg);
    el('path', { d: path(REF.map(d => [X(d[0]), Y(d[1])])), class: 'f4-line f4-ref' }, svg);
    for (const d of PLAIN) marker(svg, 'square', X(d[0]), Y(d[1]), 'f4-pt f4-plain-pt');
    for (const d of COMB) marker(svg, 'diamond', X(d[0]), Y(d[1]), 'f4-pt f4-comb-pt');
    for (const d of REF) marker(svg, 'circle', X(d[0]), Y(d[1]), 'f4-pt f4-ref-pt');
    el('text', { x: X(768), y: Y(-1.0) - 12, class: 'f4-series-label f4-ref-ink', 'text-anchor': 'middle' }, svg, 'Reference dispatch');
    el('text', { x: X(1024), y: Y(-17.5), class: 'f4-series-label f4-plain-ink', 'text-anchor': 'middle' }, svg, 'Plain dispatch');
    el('text', { x: X(256) - 14, y: Y(-25.5) + 4, class: 'f4-series-label f4-comb-ink', 'text-anchor': 'end' }, svg, 'Combine');
    for (const d of PLAIN) hit(svg, X(d[0]), Y(d[1]), `<b>Plain dispatch</b>, r<sub>d</sub>=${d[0]}<br>Combine at full width<br>Score change: <b>${pp(d[1])}</b>`);
    for (const d of COMB) hit(svg, X(d[0]), Y(d[1]), `<b>Combine</b>, r<sub>c</sub>=${d[0]}<br>Dispatch at full width<br>Task score: ${d[2].toFixed(1)}% (<b>${pp(d[1])}</b>)`);
    for (const d of REF) hit(svg, X(d[0]), Y(d[1]), `<b>Reference dispatch</b>, s=4, r<sub>d</sub>=${d[0]}<br>Combine at full width<br>Score change: <b>${pp(d[1])}</b>`);
    return H;
  });

  // ---------- (b) reference stride trade-off ----------
  EchoFit.mount(root.querySelector('[data-f4="b"]'), W => {
    const svg = root.querySelector('[data-f4="b"]');
    const H = heightFor(W), M = { l: 46, r: 16, t: 14, b: 46 };
    const X = v => M.l + (v - 10) / 32 * (W - M.l - M.r);
    const Y = v => M.t + (3 - v) / 33 * (H - M.t - M.b);
    frame(svg, W, H, M, [0, -10, -20, -30], Y, t => (t < 0 ? '−' : '') + Math.abs(t), 'Score change (pp) →', 'RDMA bytes saved (%) →');
    for (const t of [15, 25, 35]) el('text', { x: X(t), y: H - M.b + 17, class: 'f4-tick', 'text-anchor': 'middle' }, svg, t);
    const d = 2048, mBar = 2.73, mMax = 3.0;
    const saved = (s, rd) => 100 * (1 - (d * mMax / s + rd * (s - 1) / s * mBar + d * mBar) / (2 * d * mBar));
    const STRIDES = [4, 6, 8, 12], SHAPES = { 4: 'circle', 6: 'square', 8: 'diamond', 12: 'triangle' };
    const DROPS = { 1280: [0.0, 1.1, 1.0, 1.6], 1024: [0.6, 1.8, 2.3, 3.0], 768: [1.0, 2.5, 3.9, 5.3], 512: [2.6, 5.5, 6.3, 10.6], 256: [5.6, 10.7, 15.1, 20.8] };
    // label anchors copied from the paper figure's label positions (RDMA saved %, score change)
    const LABEL = { 1280: [17.35, -4.9], 1024: [23.1, -6.4], 768: [28.8, -8.7], 512: [31.5, -14.0], 256: [37.2, -24.2] };
    for (const [rd, drops] of Object.entries(DROPS)) {
      const pts = STRIDES.map((s, i) => [saved(s, +rd), -drops[i], s]);
      const cls = `f4-w${rd}`;
      el('path', { d: path(pts.map(p => [X(p[0]), Y(p[1])])), class: `f4-line ${cls}` }, svg);
      for (const p of pts) marker(svg, SHAPES[p[2]], X(p[0]), Y(p[1]), `f4-pt ${cls}-pt`, 4);
      el('text', { x: X(LABEL[rd][0]), y: Y(LABEL[rd][1]), class: 'f4-width-label', 'text-anchor': 'middle' }, svg, rd);
      for (const p of pts) hit(svg, X(p[0]), Y(p[1]), `<b>Reference dispatch</b>, s=${p[2]}, r<sub>d</sub>=${rd}<br>Combine at full width<br>RDMA saved: ${p[0].toFixed(1)}%<br>Score change: <b>${pp(p[1])}</b>`);
    }
    // stride legend
    const lg = el('g', {}, svg);
    el('text', { x: X(11), y: Y(-17.4), class: 'f4-legend-title' }, lg, 'Reference stride');
    STRIDES.forEach((s, i) => {
      const x = X(11.4) + i * 34;
      marker(lg, SHAPES[s], x, Y(-20.9), 'f4-pt f4-legend-pt', 4);
      el('text', { x: x + 8, y: Y(-20.9) + 4, class: 'f4-tick' }, lg, s);
    });
    const note = el('text', { x: X(11), y: Y(-24.6), class: 'f4-legend-note' }, lg, 'Line labels: dispatch width r');
    el('tspan', { 'baseline-shift': 'sub', 'font-size': '9' }, note, 'd');
    return H;
  });

  // ---------- (c) decode gains ----------
  EchoFit.mount(root.querySelector('[data-f4="c"]'), W => {
    const svg = root.querySelector('[data-f4="c"]');
    const H = heightFor(W), M = { l: 46, r: 46, t: 14, b: 46 };
    const REQ = [4, 32, 128, 512];
    const WORK = { 4: '128000 / 512', 32: '16384 / 512', 128: '4096 / 256', 512: '1024 / 128' };
    const X = q => M.l + 18 + REQ.indexOf(q) * ((W - M.l - M.r - 36) / 3);
    const Y = v => M.t + (1.3 - v) / 0.4 * (H - M.t - M.b);
    frame(svg, W, H, M, [0.9, 1.0, 1.1, 1.2, 1.3], Y, t => t.toFixed(1), 'Decode speedup (×) →', 'Requests per GPU');
    for (const q of REQ) el('text', { x: X(q), y: H - M.b + 17, class: 'f4-tick', 'text-anchor': 'middle' }, svg, q);
    el('line', { x1: M.l, x2: W - M.r, y1: Y(1), y2: Y(1), class: 'f4-base' }, svg);
    // [s, r_d, r_c, RDMA reduction, speedup at 4, 32, 128, 512 requests per GPU]
    const ROWS = [
      [4, 1280, 1280, 1.46, 0.941, 0.974, 0.998, 0.973],
      [6, 1280, 1280, 1.50, 0.948, 0.980, 1.013, 1.001],
      [4, 512, 1024, 2.08, 0.975, 1.002, 1.057, 1.080],
      [6, 512, 768, 2.61, 0.948, 1.017, 1.098, 1.163],
      [4, 512, 512, 2.81, 0.984, 1.016, 1.098, 1.154],
      [12, 768, 512, 2.92, 0.977, 1.027, 1.131, 1.211],
      [6, 256, 768, 3.02, 0.965, 1.023, 1.117, 1.196],
      [6, 256, 512, 3.72, 0.982, 1.030, 1.141, 1.237],
      [4, 512, 1280, 1.84, 0.927, 0.987, 1.037, 1.037, 'f4-h1'],
      [4, 512, 768, 2.39, 0.953, 1.008, 1.077, 1.120, 'f4-h2'],
      [8, 256, 512, 4.03, 0.982, 1.034, 1.154, 1.270, 'f4-h3']
    ];
    for (const r of ROWS) {
      const cls = r[8] || 'f4-other';
      const pts = REQ.map((q, i) => [X(q), Y(r[4 + i])]);
      el('path', { d: path(pts), class: `f4-line ${cls}` }, svg);
      for (const p of pts) el('circle', { cx: p[0], cy: p[1], r: r[8] ? 4 : 2.5, class: `f4-pt ${cls}-pt` }, svg);
      if (r[8]) el('text', { x: X(512) + 9, y: Y(r[7]) + 4, class: 'f4-end-value' }, svg, r[7].toFixed(2) + '×');
    }
    // legend in the empty top-left corner (low loads sit near 1.0x)
    const lg = el('g', {}, svg);
    [['f4-h3', '(8, 256, 512)'], ['f4-h2', '(4, 512, 768)'], ['f4-h1', '(4, 512, 1280)'], ['f4-other', 'others']].forEach(([cls, label], i) => {
      const y = M.t + 14 + i * 15;
      el('line', { x1: M.l + 10, x2: M.l + 26, y1: y - 4, y2: y - 4, class: `f4-line ${cls}` }, lg);
      el('text', { x: M.l + 32, y, class: 'f4-legend-label' }, lg, label);
    });
    for (const r of ROWS) REQ.forEach((q, i) => hit(svg, X(q), Y(r[4 + i]),
      `<b>EchoMoE</b> (${r[0]}, ${r[1]}, ${r[2]}), ${r[3].toFixed(2)}×<br>${q} requests per GPU (${WORK[q]} tokens)<br>Decode speedup: <b>${r[4 + i].toFixed(3)}×</b>`));
    return H;
  });
})();
