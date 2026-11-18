/* Section nav: slides in once the hero scrolls away, highlights the
   section being read, and shows reading progress. */
(function () {
  const nav = document.querySelector('.topnav');
  const hero = document.querySelector('.hero');
  if (!nav || !hero) return;

  const linkBox = nav.querySelector('.topnav-links');
  const links = Array.from(linkBox.querySelectorAll('a'));
  const sections = links
    .map(a => document.getElementById(a.getAttribute('href').slice(1)))
    .filter(Boolean);
  const progress = nav.querySelector('.topnav-progress');

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(([entry]) => {
      nav.classList.toggle('is-visible', !entry.isIntersecting);
    }, { rootMargin: '-120px 0px 0px 0px' }).observe(hero);
  } else {
    nav.classList.add('is-visible');
  }

  let active = null;
  function update() {
    const doc = document.documentElement;
    const y = window.scrollY;
    const max = doc.scrollHeight - window.innerHeight;
    if (progress) progress.style.transform = `scaleX(${max > 0 ? Math.min(1, y / max) : 0})`;

    const probe = y + window.innerHeight * 0.3;
    let current = null;
    for (const s of sections) {
      if (s.getBoundingClientRect().top + y <= probe) current = s;
    }
    // At the very bottom, the last short sections can never reach the probe line.
    if (y >= max - 2 && sections.length) current = sections[sections.length - 1];

    if (current === active) return;
    active = current;
    links.forEach(a => {
      const on = current && a.getAttribute('href') === '#' + current.id;
      a.classList.toggle('is-active', !!on);
      if (on) {
        a.setAttribute('aria-current', 'true');
        // Keep the active link in view when the link row scrolls (phones).
        if (linkBox.scrollWidth > linkBox.clientWidth) {
          linkBox.scrollTo({ left: a.offsetLeft - linkBox.clientWidth / 2 + a.offsetWidth / 2, behavior: 'smooth' });
        }
      } else {
        a.removeAttribute('aria-current');
      }
    });
  }

  let ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => { ticking = false; update(); });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  update();
})();
