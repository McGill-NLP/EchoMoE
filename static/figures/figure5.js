/* EchoMoE Figure 5: where the time goes in one prefill and one decode step
   (4096/256, 128 requests per GPU, BF16, EP32 over 4 nodes).
   Segment times (ms) are read from the paper's vector figure (results/breakdown.pdf); they sum to the
   printed step times (220.7 / 186.1 / 171.3 / 137.6 ms prefill, 40.4 / 39.6 / 37.9 / 37.0 ms decode).
   "Other" = other compute + sync wait + CPU overhead (listed separately in the tooltip). */
(() => {
  'use strict';
  const root = document.querySelector('.fig5');
  if (!root) return;
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  };

  const ROWS = ['Uncompressed', '(4, 512, 1280) · 1.84×', '(4, 512, 768) · 2.39×', '(8, 256, 512) · 4.03×'];
  // per row: comm, comp, moe, attn, other compute, sync wait, cpu overhead
  const DATA = {
    prefill: [
      [114.60, 0.00, 37.11, 27.74, 20.12, 11.67, 9.44],
      [79.37, 6.91, 35.59, 27.72, 22.24, 8.07, 6.17],
      [64.43, 6.44, 31.13, 27.70, 21.64, 13.24, 6.71],
      [42.58, 6.52, 27.57, 27.72, 19.46, 6.03, 7.68]
    ],
    decode: [
      [11.44, 0.00, 2.62, 19.19, 2.07, 1.52, 3.55],
      [9.86, 0.88, 2.23, 19.25, 2.13, 1.51, 3.73],
      [9.01, 0.84, 2.14, 19.11, 2.13, 1.01, 3.68],
      [7.33, 0.88, 1.95, 19.06, 2.05, 1.98, 3.72]
    ]
  };
  const SEGS = [
    { key: 'comm', name: 'Communication' },
    { key: 'comp', name: 'Compression' },
    { key: 'moe', name: 'MoE compute' },
    { key: 'attn', name: 'Attention compute' },
    { key: 'other', name: 'Other' }
  ];
  const fmt = v => (v >= 10 ? v.toFixed(0) : v.toFixed(1));

  const tip = root.querySelector('.f5-tip');
  function showAt(target, html) {
    tip.innerHTML = html; tip.hidden = false;
    const box = root.getBoundingClientRect(), r = target.getBoundingClientRect();
    let x = r.left - box.left + r.width / 2 - tip.offsetWidth / 2;
    x = Math.max(4, Math.min(x, box.width - tip.offsetWidth - 4));
    tip.style.left = x + 'px';
    tip.style.top = (r.top - box.top - 8) + 'px';
  }
  const hide = () => { tip.hidden = true; };

  for (const [phase, max, ticks] of [['prefill', 240, [0, 50, 100, 150, 200]], ['decode', 45, [0, 10, 20, 30, 40]]]) {
    const svg = root.querySelector(`[data-f5="${phase}"]`);
    const W = 760, H = 216, M = { l: 176, r: 128, t: 6, b: 34 };
    const X = v => M.l + v / max * (W - M.l - M.r);
    const rowH = 30, gap = 16;
    const Yr = i => M.t + i * (rowH + gap);
    for (const t of ticks) {
      el('line', { x1: X(t), x2: X(t), y1: M.t - 2, y2: H - M.b, class: 'f5-grid' }, svg);
      el('text', { x: X(t), y: H - M.b + 16, class: 'f5-tick', 'text-anchor': 'middle' }, svg, t);
    }
    el('text', { x: X(max / 2), y: H - 4, class: 'f5-axis-label', 'text-anchor': 'middle' }, svg, 'Time per step (ms)');
    const totals = DATA[phase].map(r => r.reduce((a, b) => a + b, 0));
    DATA[phase].forEach((r, i) => {
      el('text', { x: M.l - 10, y: Yr(i) + rowH / 2 + 4, class: i ? 'f5-row' : 'f5-row f5-row-base', 'text-anchor': 'end' }, svg, ROWS[i]);
      const vals = [r[0], r[1], r[2], r[3], r[4] + r[5] + r[6]];
      let acc = 0;
      vals.forEach((v, k) => {
        if (v <= 0) return;
        const x0 = X(acc), x1 = X(acc + v); acc += v;
        const seg = SEGS[k];
        const rect = el('rect', { x: x0, y: Yr(i), width: Math.max(0, x1 - x0), height: rowH, class: `f5-seg f5-${seg.key}`, tabindex: 0 }, svg);
        if (x1 - x0 >= 19) el('text', { x: (x0 + x1) / 2, y: Yr(i) + rowH / 2 + 4, class: `f5-in f5-in-${seg.key}`, 'text-anchor': 'middle' }, svg, fmt(v));
        else if (seg.key === 'comp') el('text', { x: (x0 + x1) / 2, y: Yr(i) + rowH + 11, class: 'f5-below', 'text-anchor': 'middle' }, svg, fmt(v));
        let html = `<b>${ROWS[i]}</b> · ${phase}<br>${seg.name}: <b>${v.toFixed(1)} ms</b>`;
        if (seg.key === 'other') html += `<br><span class="f5-sub">Other compute ${r[4].toFixed(1)} · Sync wait ${r[5].toFixed(1)} · CPU overhead ${r[6].toFixed(1)} ms</span>`;
        if (seg.key === 'comm' && i) html += `<br><span class="f5-sub">${(100 * (1 - v / DATA[phase][0][0])).toFixed(0)}% less than uncompressed (${DATA[phase][0][0].toFixed(1)} ms)</span>`;
        rect.addEventListener('mouseenter', () => showAt(rect, html));
        rect.addEventListener('mouseleave', hide);
        rect.addEventListener('focus', () => showAt(rect, html));
        rect.addEventListener('blur', hide);
      });
      const tot = totals[i];
      const label = i ? `${tot.toFixed(1)} ms · ${(totals[0] / tot).toFixed(2)}×` : `${tot.toFixed(1)} ms`;
      el('text', { x: X(tot) + 8, y: Yr(i) + rowH / 2 + 4, class: i ? 'f5-total f5-total-echo' : 'f5-total' }, svg, label);
    });
  }
})();
