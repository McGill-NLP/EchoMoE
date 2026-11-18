/* Results charts (Figs. 3-6) draw themselves in the first time they scroll
   into view: a left-to-right sweep reveals lines, areas and bars, while
   points pop and labels fade in as the sweep passes them. Plays once per
   chart. Skipped entirely with prefers-reduced-motion. Charts that redraw
   at a new width (see fit.js) re-arm if they have not played yet. */
(function () {
  if (!('IntersectionObserver' in window)) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const NS = 'http://www.w3.org/2000/svg';
  // Chart frame, shown immediately: axes, grid, ticks, baselines, legends, hit areas.
  const FRAME = new Set(['grid', 'tick', 'axis', 'axis-label', 'base', 'note', 'hit', 'guide',
    'hover-guide', 'row', 'row-base', 'legend-title', 'legend-note', 'legend-pt']);
  const tokens = el => (el.getAttribute('class') || '').split(/\s+/).map(t => t.replace(/^f\d-/, ''));
  const easeInOut = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const linear = t => t;
  const bySvg = new Map();
  let uid = 0;

  function arm(svg, opts) {
    const sweep = [], pending = [];
    let x0 = Infinity, x1 = -Infinity;
    for (const el of svg.querySelectorAll('path, line, circle, rect, polyline, polygon, text')) {
      if (el.closest('defs, clipPath, marker')) continue;
      const t = tokens(el);
      if (t.some(k => FRAME.has(k))) continue;
      const b = el.getBBox();
      if (!b.width && !b.height) continue;
      if (el.tagName === 'text') {
        pending.push({ el, at: b.x + b.width });          // fade in once fully passed
        el.classList.add('rv-fade');
      } else if (t.some(k => k === 'pt' || k.endsWith('-pt') || k === 'ring')) {
        pending.push({ el, at: b.x + b.width / 2 });      // pop when the sweep hits its center
        el.classList.add('rv-pop');
        x0 = Math.min(x0, b.x); x1 = Math.max(x1, b.x + b.width);
      } else {
        sweep.push(el);
        x0 = Math.min(x0, b.x); x1 = Math.max(x1, b.x + b.width);
      }
    }
    if (!isFinite(x0)) return null;
    x0 -= 4; x1 += 4;
    const vb = svg.viewBox.baseVal;
    const clip = document.createElementNS(NS, 'clipPath');
    clip.id = 'rv-clip-' + (++uid);
    clip.setAttribute('clipPathUnits', 'userSpaceOnUse');
    const rect = document.createElementNS(NS, 'rect');
    rect.setAttribute('x', x0); rect.setAttribute('y', vb.y - 20);
    rect.setAttribute('width', 0); rect.setAttribute('height', vb.height + 40);
    clip.appendChild(rect);
    svg.insertBefore(clip, svg.firstChild);
    sweep.forEach(el => el.setAttribute('clip-path', `url(#${clip.id})`));
    pending.sort((a, b) => a.at - b.at);
    svg.classList.add('rv-armed');
    return { svg, sweep, pending, clip, rect, x0, x1, done: false, ...opts };
  }

  function finish(c) {
    if (c.done) return;
    c.done = true;
    c.pending.forEach(p => p.el.classList.add('rv-in'));
    c.sweep.forEach(el => el.removeAttribute('clip-path'));
    c.clip.remove();
    // let the last fades finish, then drop the reveal styling entirely
    setTimeout(() => c.svg.classList.remove('rv-armed'), 600);
  }

  function play(c, delay) {
    let start = null, next = 0;
    const span = c.x1 - c.x0;
    function frame(now) {
      if (c.done) return;
      if (start === null) start = now + delay;
      const t = Math.max(0, Math.min(1, (now - start) / c.duration));
      const x = c.x0 + span * c.ease(t);
      c.rect.setAttribute('width', x - c.x0);
      while (next < c.pending.length && c.pending[next].at <= x) c.pending[next++].el.classList.add('rv-in');
      if (t < 1) requestAnimationFrame(frame); else finish(c);
    }
    requestAnimationFrame(frame);
  }

  function init() {
    const specs = [
      ['.fig3 .f3-chart', { duration: 1500, ease: easeInOut }],
      ['.fig4 .f4-chart', { duration: 1400, ease: easeInOut }],
      ['.fig5 .f5-chart', { duration: 1600, ease: linear }],   // linear: bars that finish first are faster
      ['.fig6 .f6-chart', { duration: 1700, ease: easeInOut }]
    ];
    const safeArm = (svg, opts) => {
      try { return arm(svg, opts); } catch (e) { svg.classList.remove('rv-armed'); return null; }
    };
    for (const [sel, opts] of specs) {
      for (const svg of document.querySelectorAll(sel)) {
        const c = safeArm(svg, opts);
        if (c) bySvg.set(svg, c);
      }
    }
    const io = new IntersectionObserver(entries => {
      let k = 0;
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        io.unobserve(e.target);
        const c = bySvg.get(e.target);
        if (!c) continue;
        c.played = true;
        play(c, 150 * k++);   // side-by-side panels cascade
      }
    }, { threshold: 0.35 });
    bySvg.forEach(c => io.observe(c.svg));

    // A redraw replaces every mark: re-arm a chart still waiting to play;
    // one that already played (or is playing) just shows its new marks.
    document.addEventListener('chart:render', e => {
      const old = bySvg.get(e.target);
      if (!old) return;
      if (old.played) {
        old.done = true;
        old.svg.classList.remove('rv-armed');
        return;
      }
      const c = safeArm(old.svg, { duration: old.duration, ease: old.ease });
      if (c) bySvg.set(old.svg, c);
      else { bySvg.delete(old.svg); io.unobserve(old.svg); }
    });
    window.addEventListener('beforeprint', () => bySvg.forEach(finish));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
