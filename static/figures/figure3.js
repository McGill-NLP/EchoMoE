/* EchoMoE Figure 3: task score and throughput vs. inter-node RDMA saved.
   All numbers are taken from the paper (Qwen3-30B-A3B, full-model adaptation):
   - RDMA reduction, overall task score and delta: Table "Task performance for all adapted joint configurations".
   - Throughput: Table "Throughput and latency across workloads", 4096/256 (128 requests per GPU) column.
   RDMA saved (%) = 100 * (1 - 1 / reduction). */
(() => {
  'use strict';
  const root = document.querySelector('.fig3');
  if (!root) return;
  const NS = 'http://www.w3.org/2000/svg';

  // [s, r_d, r_c, RDMA reduction (x), overall score (%), delta (pp), throughput speedup (x), highlighted]
  const ECHO = [
    [4, 1280, 1280, 1.46, 65.5, -1.4, 1.025, false],
    [6, 1280, 1280, 1.50, 65.1, -1.8, 1.055, false],
    [4, 512, 1280, 1.84, 64.9, -2.0, 1.115, true],
    [4, 512, 1024, 2.08, 63.8, -3.1, 1.169, false],
    [4, 512, 768, 2.39, 63.1, -3.8, 1.226, true],
    [6, 512, 768, 2.61, 61.5, -5.4, 1.276, false],
    [4, 512, 512, 2.81, 60.6, -6.4, 1.287, false],
    [12, 768, 512, 2.92, 59.4, -7.6, 1.348, false],
    [6, 256, 768, 3.02, 58.3, -8.6, 1.325, false],
    [6, 256, 512, 3.72, 56.9, -10.1, 1.396, false],
    [8, 256, 512, 4.03, 55.7, -11.2, 1.429, true]
  ];
  // [r_d, r_c, RDMA reduction (x), overall score (%), delta (pp)]
  const PLAIN = [
    [1280, 1024, 1.78, 62.8, -4.1],
    [1024, 1024, 2.00, 61.6, -5.3],
    [768, 768, 2.67, 58.0, -8.9],
    [512, 512, 4.00, 51.4, -15.5]
  ];
  const BASE = 66.9;
  const saved = red => 100 * (1 - 1 / red);
  const fmtRed = red => red.toFixed(2) + '×';
  const fmtDelta = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(1) + ' pp';
  const gainOf = sp => +(100 * (sp - 1)).toFixed(1);

  // Plotting rows: [RDMA saved %, task score %, throughput gain %, label-or-null, source row]
  const echo = ECHO.map(r => [saved(r[3]), r[4], gainOf(r[6]), r[7] ? fmtRed(r[3]) : null, r]);
  const plain = PLAIN.map(r => [saved(r[2]), r[3], null, null, r]);

  const el = (tag, attrs, parent, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  };

  const tip = root.querySelector('.f3-tip');
  function showTip(evt, html) {
    tip.innerHTML = html;
    tip.hidden = false;
    const box = root.getBoundingClientRect();
    let x = evt.clientX - box.left + 14, y = evt.clientY - box.top - 12;
    const w = tip.offsetWidth;
    if (x + w > box.width - 6) x = evt.clientX - box.left - w - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  const hideTip = () => { tip.hidden = true; };

  // Layout for a panel drawn at its real pixel width W.
  function layout(W) {
    const H = Math.round(Math.max(270, Math.min(300, W * 0.75)));
    // Leave room right of the last highlighted point (75.2% saved) for its value
    // label; both panels share this margin so their x-axes stay aligned.
    const k = (80 - 75.2) / 52;
    const r = Math.max(14, Math.ceil((72 - k * (W - 46)) / (1 - k)));
    const M = { l: 46, r, t: 16, b: 44 };
    const X = v => M.l + (v - 28) / (80 - 28) * (W - M.l - M.r);
    const mkY = (lo, hi) => v => M.t + (hi - v) / (hi - lo) * (H - M.t - M.b);
    return { W, H, M, X, mkY };
  }

  function axes(svg, L, Y, ticks, yLabel, fmt) {
    const { W, H, M, X } = L;
    const g = el('g', { class: 'f3-axes' }, svg);
    for (const t of ticks) {
      el('line', { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), class: 'f3-grid' }, g);
      el('text', { x: M.l - 8, y: Y(t) + 4, class: 'f3-tick', 'text-anchor': 'end' }, g, fmt(t));
    }
    for (const t of [30, 40, 50, 60, 70, 80]) {
      el('text', { x: X(t), y: H - M.b + 17, class: 'f3-tick', 'text-anchor': 'middle' }, g, t);
    }
    el('line', { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: 'f3-axis' }, g);
    el('text', { x: (M.l + W - M.r) / 2, y: H - 8, class: 'f3-axis-label', 'text-anchor': 'middle' }, g, 'Inter-node RDMA saved (%) →');
    el('text', { x: 13, y: (M.t + H - M.b) / 2, class: 'f3-axis-label', 'text-anchor': 'middle',
      transform: `rotate(-90 13 ${(M.t + H - M.b) / 2})` }, g, yLabel);
    // guides at the highlighted configurations
    for (const d of echo.filter(d => d[3])) {
      el('line', { x1: X(d[0]), x2: X(d[0]), y1: M.t, y2: H - M.b, class: 'f3-guide' }, g);
    }
  }

  const path = (pts, X, Y) => pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');

  function points(svg, X, data, Y, valIdx, cls, tipFn) {
    const g = el('g', {}, svg);
    for (const d of data) {
      const cx = X(d[0]), cy = Y(d[valIdx]);
      if (cls === 'f3-plain-pt') el('rect', { x: cx - 4.5, y: cy - 4.5, width: 9, height: 9, rx: 1.5, class: cls }, g);
      else {
        if (d[3]) el('circle', { cx, cy, r: 8.5, class: 'f3-ring' }, g);
        el('circle', { cx, cy, r: 4.5, class: cls }, g);
      }
      const hit = el('circle', { cx, cy, r: 13, class: 'f3-hit', tabindex: 0 }, g);
      hit.addEventListener('mouseenter', e => showTip(e, tipFn(d)));
      hit.addEventListener('mousemove', e => showTip(e, tipFn(d)));
      hit.addEventListener('mouseleave', hideTip);
      hit.addEventListener('focus', () => {
        const r = hit.getBoundingClientRect();
        showTip({ clientX: r.left + r.width / 2, clientY: r.top }, tipFn(d));
      });
      hit.addEventListener('blur', hideTip);
    }
  }

  const echoTip = d => { const r = d[4]; return `<b>EchoMoE</b> (s=${r[0]}, r<sub>d</sub>=${r[1]}, r<sub>c</sub>=${r[2]})<br>
    RDMA reduction: ${fmtRed(r[3])} (${Math.round(d[0])}% saved)<br>
    Task score: ${r[4].toFixed(1)}% (${fmtDelta(r[5])})<br>
    Throughput: ${r[6].toFixed(3)}× (+${d[2].toFixed(1)}%)`; };
  const plainTip = d => { const r = d[4]; return `<b>Plain compression</b> (r<sub>d</sub>=${r[0]}, r<sub>c</sub>=${r[1]})<br>
    RDMA reduction: ${fmtRed(r[2])} (${Math.round(d[0])}% saved)<br>
    Task score: ${r[3].toFixed(1)}% (${fmtDelta(r[4])})`; };

  // ---- Left: task score ----
  EchoFit.mount(root.querySelector('[data-f3="score"]'), W => {
    const svg = root.querySelector('[data-f3="score"]');
    const L = layout(W), { H, M, X } = L;
    const Y = L.mkY(50, 68);
    axes(svg, L, Y, [50, 55, 60, 65], 'Task score (%) →', t => t);
    el('line', { x1: M.l, x2: W - M.r, y1: Y(BASE), y2: Y(BASE), class: 'f3-base' }, svg);
    el('text', { x: W - M.r - 4, y: Y(BASE) - 6, class: 'f3-note', 'text-anchor': 'end' }, svg, 'Uncompressed (66.9)');
    // shaded gap between the two frontiers
    const x0 = plain[0][0];
    const i = echo.findIndex(d => d[0] >= x0), a = echo[i - 1], b = echo[i];
    const yAt = a[1] + (b[1] - a[1]) * (x0 - a[0]) / (b[0] - a[0]);
    const gap = echo.slice(i).map(d => [d[0], d[1]]);
    el('path', { d: path([[x0, yAt], ...gap], X, Y) + path([...plain].reverse(), X, Y).replace('M', 'L') + 'Z', class: 'f3-gap' }, svg);
    el('path', { d: path(plain, X, Y), class: 'f3-plain-line' }, svg);
    el('path', { d: path(echo.map(d => [d[0], d[1]]), X, Y), class: 'f3-echo-line' }, svg);
    // arrow: same score, fewer bytes
    // plain (1.78x, 62.8) vs EchoMoE (2.39x, 63.1): 1 - 1.78/2.39 = 26% fewer bytes
    const pA = plain[0], eB = echo.find(d => d[4][3] === 2.39);
    el('line', { x1: X(pA[0]) + 8, y1: Y(pA[1]), x2: X(eB[0]) - 12, y2: Y(pA[1]), class: 'f3-arrow', 'marker-end': 'url(#f3-head)' }, svg);
    // callout in the empty lower-left corner: one line if it clears the
    // baseline's line, otherwise stacked
    const c1 = el('text', { x: X(29.5), y: Y(58.6), class: 'f3-callout' }, svg, '26% fewer RDMA bytes');
    if (X(29.5) + EchoFit.textWidth(c1) < X(57.3) - 6) {
      el('text', { x: X(29.5), y: Y(58.6) + 15, class: 'f3-callout-sub' }, svg, 'at similar performance');
    } else {
      const cx = M.l + 8, cy = Y(57.2);
      c1.setAttribute('x', cx); c1.setAttribute('y', cy); c1.textContent = '26% fewer';
      el('text', { x: cx, y: cy + 15, class: 'f3-callout' }, svg, 'RDMA bytes');
      el('text', { x: cx, y: cy + 30, class: 'f3-callout-sub' }, svg, 'at similar performance');
    }
    // line names: EchoMoE under the start of its line, the baseline beside its last point
    const lab = el('text', { x: X(29.5), y: Y(63.5), class: 'f3-label-echo' }, svg, 'EchoMoE');
    if (X(29.5) + EchoFit.textWidth(lab) > X(echo[2][0]) - 12) {   // too tight: sit above the line's right end
      lab.setAttribute('x', W - M.r); lab.setAttribute('y', Y(61.8)); lab.setAttribute('text-anchor', 'end');
    }
    el('text', { x: X(75) - 10, y: H - M.b - 6, class: 'f3-label-plain', 'text-anchor': 'end' }, svg, 'Plain compression');
    for (const d of echo.filter(d => d[3])) {
      const t = el('text', { x: X(d[0]) + 11, y: Y(d[1]) - 9, class: 'f3-value' }, svg, fmtDelta(d[4][5]));
      if (X(d[0]) + 11 + EchoFit.textWidth(t) > W - 2) {   // no room on the right: sit below the point
        t.setAttribute('x', X(d[0]) + 4); t.setAttribute('y', Y(d[1]) + 24); t.setAttribute('text-anchor', 'end');
      }
    }
    points(svg, X, plain, Y, 1, 'f3-plain-pt', plainTip);
    points(svg, X, echo, Y, 1, 'f3-echo-pt', echoTip);
    return H;
  });

  // ---- Right: throughput gain ----
  EchoFit.mount(root.querySelector('[data-f3="throughput"]'), W => {
    const svg = root.querySelector('[data-f3="throughput"]');
    const L = layout(W), { H, M, X } = L;
    const Y = L.mkY(-6, 50);
    axes(svg, L, Y, [0, 10, 20, 30, 40, 50], 'Throughput gain (%) →', t => t);
    el('line', { x1: M.l, x2: W - M.r, y1: Y(0), y2: Y(0), class: 'f3-base' }, svg);
    el('text', { x: M.l + 6, y: Y(0) + 16, class: 'f3-note' }, svg, 'Uncompressed (0%)');
    const line = echo.map(d => [d[0], d[2]]);
    el('path', { d: path(line, X, Y) + `L${X(line[line.length - 1][0])},${Y(0)}L${X(line[0][0])},${Y(0)}Z`, class: 'f3-area' }, svg);
    el('path', { d: path(line, X, Y), class: 'f3-echo-line' }, svg);
    for (const d of echo.filter(d => d[3])) {
      el('text', { x: X(d[0]) - 12, y: Y(d[2]) - 8, class: 'f3-value', 'text-anchor': 'end' }, svg, `+${d[2].toFixed(1)}%`);
    }
    points(svg, X, echo, Y, 2, 'f3-echo-pt', echoTip);
    return H;
  });
})();
