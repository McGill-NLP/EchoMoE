/* EchoMoE Figure 6: throughput speedup over TCP (Ethernet, ~23 Gb/s per node).
   Per-configuration speedups are read from the paper's vector figure (results/throughput_tcp.pdf,
   4096/256 with 8 requests per GPU) and rounded to 2 decimals; the end points match Table
   "Throughput speedup across networks" (BF16 1.49-2.38x, FP8 1.86-3.24x).
   RDMA reductions come from the adapted-configuration table; RDMA saved (%) = 100 * (1 - 1 / reduction). */
(() => {
  'use strict';
  const root = document.querySelector('.fig6');
  if (!root) return;
  const NS = 'http://www.w3.org/2000/svg';

  // [s, r_d, r_c, RDMA reduction, TCP BF16, TCP FP8, InfiniBand BF16]
  const ROWS = [
    [4, 1280, 1280, 1.46, 1.49, 1.87, 0.96],
    [6, 1280, 1280, 1.50, 1.52, 1.86, 0.97],
    [4, 512, 1280, 1.84, 1.69, 2.06, 1.00],
    [4, 512, 1024, 2.08, 1.83, 2.14, 1.01],
    [4, 512, 768, 2.39, 1.93, 2.16, 1.04],
    [6, 512, 768, 2.61, 1.97, 2.26, 1.06],
    [4, 512, 512, 2.81, 2.10, 2.28, 1.07],
    [12, 768, 512, 2.92, 2.11, 2.20, 1.08],
    [6, 256, 768, 3.02, 2.09, 3.00, 1.08],
    [6, 256, 512, 3.72, 2.32, 3.12, 1.09],
    [8, 256, 512, 4.03, 2.38, 3.24, 1.10]
  ];
  const SERIES = [
    { key: 'fp8', idx: 5, name: 'TCP, FP8', cls: 'f6-fp8' },
    { key: 'bf16', idx: 4, name: 'TCP, BF16', cls: 'f6-bf16' },
    { key: 'ib', idx: 6, name: 'InfiniBand, BF16', cls: 'f6-ib' }
  ];
  const saved = red => 100 * (1 - 1 / red);

  const el = (tag, attrs, parent, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  };
  const svg = root.querySelector('svg.f6-chart');
  const tip = root.querySelector('.f6-tip');
  const pts = ROWS.map(r => [saved(r[3]), r]);

  EchoFit.mount(svg, W => {
    // narrow screens: short end labels (the legend above names them in full)
    const narrow = W < 520;
    const names = narrow ? { fp8: 'FP8', bf16: 'BF16', ib: 'IB' } : null;
    const H = Math.round(Math.max(270, Math.min(330, W * 0.46)));
    const probe = SERIES.map(s => el('text', { class: `f6-end ${s.cls}-ink` }, svg, names ? names[s.key] : s.name));
    const endW = Math.max(...probe.map(EchoFit.textWidth));
    probe.forEach(t => t.remove());
    const M = { l: 52, r: Math.ceil(endW) + 26, t: 20, b: 48 };
    const X = v => M.l + (v - 28) / (80 - 28) * (W - M.l - M.r);
    const Y = v => M.t + (3.4 - v) / (3.4 - 0.7) * (H - M.t - M.b);

    // axes and grid
    for (const t of [1, 1.5, 2, 2.5, 3]) {
      el('line', { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), class: 'f6-grid' }, svg);
      el('text', { x: M.l - 8, y: Y(t) + 4, class: 'f6-tick', 'text-anchor': 'end' }, svg, t.toFixed(1) + '×');
    }
    for (const t of [30, 40, 50, 60, 70, 80]) {
      el('text', { x: X(t), y: H - M.b + 17, class: 'f6-tick', 'text-anchor': 'middle' }, svg, t);
    }
    el('line', { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: 'f6-axis' }, svg);
    el('text', { x: (M.l + W - M.r) / 2, y: H - 8, class: 'f6-axis-label', 'text-anchor': 'middle' }, svg, 'Inter-node RDMA saved (%) →');
    const midY = (M.t + H - M.b) / 2;
    el('text', { x: 13, y: midY, class: 'f6-axis-label', 'text-anchor': 'middle', transform: `rotate(-90 13 ${midY})` }, svg, 'Throughput speedup (×) →');
    // baseline stops at the last configuration so the end labels sit clear of it
    el('line', { x1: M.l, x2: X(pts[pts.length - 1][0]) + 8, y1: Y(1), y2: Y(1), class: 'f6-base' }, svg);
    el('text', { x: X(35.5), y: Y(1) + 15, class: 'f6-note' }, svg, 'Uncompressed (1.0×)');

    // lines, markers, end labels
    for (const s of SERIES) {
      const d = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1][s.idx]).toFixed(1)}`).join('');
      el('path', { d, class: `f6-line ${s.cls}` }, svg);
      for (const p of pts) {
        const cx = X(p[0]), cy = Y(p[1][s.idx]);
        if (s.key === 'fp8') el('rect', { x: cx - 4.5, y: cy - 4.5, width: 9, height: 9, rx: 1.5, class: `f6-pt ${s.cls}` }, svg);
        else el('circle', { cx, cy, r: s.key === 'ib' ? 3.5 : 4.5, class: `f6-pt ${s.cls}` }, svg);
      }
      const last = pts[pts.length - 1];
      el('text', { x: X(last[0]) + 14, y: Y(last[1][s.idx]) + 5, class: `f6-end ${s.cls}-ink` }, svg, names ? names[s.key] : s.name);
    }
    // peak values
    const last = pts[pts.length - 1];
    el('text', { x: X(last[0]), y: Y(last[1][5]) - 13, class: 'f6-value', 'text-anchor': 'middle' }, svg, `${last[1][5].toFixed(2)}×`);
    el('text', { x: X(last[0]), y: Y(last[1][4]) - 13, class: 'f6-value', 'text-anchor': 'middle' }, svg, `${last[1][4].toFixed(2)}×`);

    // hover: one column per configuration
    const guide = el('line', { y1: M.t, y2: H - M.b, class: 'f6-hover-guide', visibility: 'hidden' }, svg);
    const xs = pts.map(p => X(p[0]));
    pts.forEach((p, i) => {
      const left = i ? (xs[i - 1] + xs[i]) / 2 : xs[i] - 12;
      const right = i < xs.length - 1 ? (xs[i] + xs[i + 1]) / 2 : xs[i] + 12;
      const hit = el('rect', { x: left, y: M.t, width: right - left, height: H - M.t - M.b, class: 'f6-hit', tabindex: 0 }, svg);
      const r = p[1];
      const html = `<b>EchoMoE</b> (s=${r[0]}, r<sub>d</sub>=${r[1]}, r<sub>c</sub>=${r[2]})<br>
        RDMA reduction: ${r[3].toFixed(2)}× (${Math.round(p[0])}% saved)<br>
        <span class="f6-sw f6-fp8"></span>TCP, FP8: <b>${r[5].toFixed(2)}×</b><br>
        <span class="f6-sw f6-bf16"></span>TCP, BF16: <b>${r[4].toFixed(2)}×</b><br>
        <span class="f6-sw f6-ib"></span>InfiniBand, BF16: <b>${r[6].toFixed(2)}×</b>`;
      const show = () => {
        guide.setAttribute('x1', xs[i]); guide.setAttribute('x2', xs[i]); guide.setAttribute('visibility', 'visible');
        tip.innerHTML = html; tip.hidden = false;
        const box = root.getBoundingClientRect(), sbox = svg.getBoundingClientRect();
        const px = sbox.left - box.left + xs[i] / W * sbox.width;
        let x = px + 14;
        if (x + tip.offsetWidth > box.width - 6) x = px - tip.offsetWidth - 14;
        tip.style.left = Math.max(4, x) + 'px';
        tip.style.top = (sbox.top - box.top + Y(r[5]) / H * sbox.height) + 'px';
      };
      const hide = () => { tip.hidden = true; guide.setAttribute('visibility', 'hidden'); };
      hit.addEventListener('mouseenter', show);
      hit.addEventListener('mouseleave', hide);
      hit.addEventListener('focus', show);
      hit.addEventListener('blur', hide);
    });
    return H;
  });
})();
