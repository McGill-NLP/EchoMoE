/* Responsive drawing for the results charts (Figs. 3-6).
   Each chart is redrawn at its real on-screen width, so one SVG unit is one
   CSS pixel: text keeps the same size on a phone and on a desktop, and only
   the plot area stretches. Charts redraw when their width changes and fire a
   bubbling `chart:render` event so the entrance animation can re-arm. */
(function () {
  'use strict';
  const charts = [];

  function mount(svg, draw) {
    if (!svg) return;
    const c = { svg, draw, w: 0 };
    c.render = force => {
      const w = Math.round(svg.getBoundingClientRect().width);
      if (!w || (!force && Math.abs(w - c.w) < 2)) return;
      c.w = w;
      for (const n of [...svg.childNodes]) if (n.nodeName !== 'defs') n.remove();
      const h = Math.round(draw(w));
      svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      svg.dispatchEvent(new CustomEvent('chart:render', { bubbles: true }));
    };
    c.render();
    charts.push(c);
    if ('ResizeObserver' in window) {
      let raf = 0;
      new ResizeObserver(() => {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => c.render());
      }).observe(svg);
    }
  }

  // Labels are measured while drawing; redraw once the web font has arrived.
  if (document.fonts && document.fonts.status !== 'loaded') {
    document.fonts.ready.then(() => charts.forEach(c => c.render(true)));
  }

  // Width of a text element already in the DOM (falls back to a rough estimate).
  function textWidth(t) {
    try { return t.getComputedTextLength(); } catch (e) { return t.textContent.length * 7; }
  }

  window.EchoFit = { mount, textWidth };
})();
