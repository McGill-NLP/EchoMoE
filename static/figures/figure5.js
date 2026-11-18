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

  // Wide screens: row labels left of the bars, totals right of them.
  // Narrow screens: each row's label and total sit on a line above its bar.
  for (const [phase, max, ticks] of [['prefill', 240, [0, 50, 100, 150, 200]], ['decode', 45, [0, 10, 20, 30, 40]]]) {
    const svg = root.querySelector(`[data-f5="${phase}"]`);
    EchoFit.mount(svg, W => {
      const rows = DATA[phase];
      const totals = rows.map(r => r.reduce((a, b) => a + b, 0));
      const totalText = i => (i ? `${totals[i].toFixed(1)} ms · ${(totals[0] / totals[i]).toFixed(2)}×` : `${totals[i].toFixed(1)} ms`);
      // measure the labels to size the margins
      const probe = rows.map((r, i) => [
        el('text', { class: i ? 'f5-row' : 'f5-row f5-row-base' }, svg, ROWS[i]),
        el('text', { class: i ? 'f5-total f5-total-echo' : 'f5-total' }, svg, totalText(i))
      ]);
      const labelW = Math.max(...probe.map(p => EchoFit.textWidth(p[0])));
      const totalW = Math.max(...probe.map(p => EchoFit.textWidth(p[1])));
      probe.flat().forEach(t => t.remove());
      const narrow = W - labelW - totalW - 28 < 300;
      // wide rows leave room under each bar for the small compression value
      const rowH = narrow ? 24 : 26, gap = narrow ? 12 : 18, head = narrow ? 19 : 0;
      const M = narrow ? { l: 4, r: 4, t: 2, b: 36 } : { l: Math.ceil(labelW) + 12, r: Math.ceil(totalW) + 14, t: 2, b: 36 };
      const X = v => M.l + v / max * (W - M.l - M.r);
      const Yr = i => M.t + head + i * (head + rowH + gap);   // top of bar i
      const axisY = Yr(rows.length - 1) + rowH + (narrow ? 6 : 16);
      const H = axisY + M.b;
      for (const t of ticks) {
        el('line', { x1: X(t), x2: X(t), y1: M.t, y2: axisY, class: 'f5-grid' }, svg);
        el('text', { x: X(t), y: axisY + 15, class: 'f5-tick', 'text-anchor': 'middle' }, svg, t);
      }
      el('text', { x: X(max / 2), y: H - 4, class: 'f5-axis-label', 'text-anchor': 'middle' }, svg, 'Time per step (ms)');
      rows.forEach((r, i) => {
        const y = Yr(i), mid = y + rowH / 2 + 4;
        if (narrow) {
          el('text', { x: M.l, y: y - 6, class: i ? 'f5-row' : 'f5-row f5-row-base' }, svg, ROWS[i]);
          el('text', { x: W - M.r, y: y - 6, class: i ? 'f5-total f5-total-echo' : 'f5-total', 'text-anchor': 'end' }, svg, totalText(i));
        } else {
          el('text', { x: M.l - 10, y: mid, class: i ? 'f5-row' : 'f5-row f5-row-base', 'text-anchor': 'end' }, svg, ROWS[i]);
        }
        const vals = [r[0], r[1], r[2], r[3], r[4] + r[5] + r[6]];
        let acc = 0;
        vals.forEach((v, k) => {
          if (v <= 0) return;
          const x0 = X(acc), x1 = X(acc + v); acc += v;
          const seg = SEGS[k];
          const rect = el('rect', { x: x0, y, width: Math.max(0, x1 - x0), height: rowH, class: `f5-seg f5-${seg.key}`, tabindex: 0 }, svg);
          const num = el('text', { x: (x0 + x1) / 2, y: mid, class: `f5-in f5-in-${seg.key}`, 'text-anchor': 'middle' }, svg, fmt(v));
          if (EchoFit.textWidth(num) + 8 > x1 - x0) {
            num.remove();
            // compression is the one small segment worth naming: label it just below its bar
            if (seg.key === 'comp' && !narrow) el('text', { x: (x0 + x1) / 2, y: y + rowH + 13, class: 'f5-below', 'text-anchor': 'middle' }, svg, fmt(v));
          }
          let html = `<b>${ROWS[i]}</b> · ${phase}<br>${seg.name}: <b>${v.toFixed(1)} ms</b>`;
          if (seg.key === 'other') html += `<br><span class="f5-sub">Other compute ${r[4].toFixed(1)} · Sync wait ${r[5].toFixed(1)} · CPU overhead ${r[6].toFixed(1)} ms</span>`;
          if (seg.key === 'comm' && i) html += `<br><span class="f5-sub">${(100 * (1 - v / rows[0][0])).toFixed(0)}% less than uncompressed (${rows[0][0].toFixed(1)} ms)</span>`;
          rect.addEventListener('mouseenter', () => showAt(rect, html));
          rect.addEventListener('mouseleave', hide);
          rect.addEventListener('focus', () => showAt(rect, html));
          rect.addEventListener('blur', hide);
        });
        if (!narrow) el('text', { x: X(totals[i]) + 8, y: mid, class: i ? 'f5-total f5-total-echo' : 'f5-total' }, svg, totalText(i));
      });
      return H;
    });
  }
})();
