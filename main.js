/* ===================================================
   RideRush — main.js
   Interactions, countdown timers, ML prediction,
   scroll reveal, chart generation
   =================================================== */

(function () {
  'use strict';

  /* ==========================
     NAVBAR scroll behavior
     ========================== */
  const navbar = document.getElementById('navbar');
  const onScroll = () => navbar.classList.toggle('scrolled', window.scrollY > 40);
  window.addEventListener('scroll', onScroll, { passive: true });

  /* ==========================
     HAMBURGER
     ========================== */
  const hamburger = document.getElementById('hamburger');
  const mobileMenu = document.getElementById('mobile-menu');
  if (hamburger) {
    hamburger.addEventListener('click', () => mobileMenu.classList.toggle('open'));
    mobileMenu.querySelectorAll('a').forEach(a => a.addEventListener('click', () => mobileMenu.classList.remove('open')));
  }

  /* ==========================
     HERO BG IMAGE ZOOM
     ========================== */
  const heroBgImg = document.getElementById('hero-bg-img');
  if (heroBgImg) {
    if (heroBgImg.complete) heroBgImg.classList.add('loaded');
    else heroBgImg.addEventListener('load', () => heroBgImg.classList.add('loaded'));
  }

  /* ==========================
     HERO TITLE CYCLER
     ========================== */
  const heroTitles = ['Manhattan', 'Brooklyn', 'Queens', 'The Bronx', 'Staten Island'];
  let titleIdx = 0;
  const heroTitle = document.getElementById('hero-title');
  if (heroTitle) {
    setInterval(() => {
      heroTitle.style.opacity = '0';
      heroTitle.style.transform = 'translateY(16px)';
      setTimeout(() => {
        titleIdx = (titleIdx + 1) % heroTitles.length;
        heroTitle.textContent = heroTitles[titleIdx];
        heroTitle.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        heroTitle.style.opacity = '1';
        heroTitle.style.transform = 'translateY(0)';
      }, 350);
    }, 3200);
  }

  /* ==========================
     COUNTDOWN TIMERS
     ========================== */
  const timerEls = document.querySelectorAll('.timer-val');
  timerEls.forEach(el => {
    let secs = parseInt(el.getAttribute('data-seconds'), 10);
    const tick = () => {
      if (secs <= 0) { el.textContent = '00:00:00'; return; }
      secs--;
      const h = Math.floor(secs / 3600).toString().padStart(2, '0');
      const m = Math.floor((secs % 3600) / 60).toString().padStart(2, '0');
      const s = (secs % 60).toString().padStart(2, '0');
      el.textContent = `${h}:${m}:${s}`;
    };
    tick();
    setInterval(tick, 1000);
  });

  /* ==========================
     DEALS TRACK ARROWS
     ========================== */
  const track = document.getElementById('deals-track');
  const prevBtn = document.getElementById('track-prev');
  const nextBtn = document.getElementById('track-next');
  if (track && prevBtn && nextBtn) {
    const SCROLL_AMOUNT = 280;
    nextBtn.addEventListener('click', () => track.scrollBy({ left: SCROLL_AMOUNT, behavior: 'smooth' }));
    prevBtn.addEventListener('click', () => track.scrollBy({ left: -SCROLL_AMOUNT, behavior: 'smooth' }));
  }

  /* ==========================
     ANIMATED COUNTERS
     ========================== */
  function animateCounter(el) {
    const target = parseInt(el.getAttribute('data-count'), 10);
    const duration = 1800;
    const step = 16;
    const inc = target / (duration / step);
    let current = 0;
    const timer = setInterval(() => {
      current += inc;
      if (current >= target) { current = target; clearInterval(timer); }
      el.textContent = Math.floor(current).toLocaleString();
    }, step);
  }

  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        counterObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });
  document.querySelectorAll('[data-count]').forEach(el => counterObserver.observe(el));

  /* ==========================
     REVEAL ANIMATIONS
     ========================== */
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  // Automatically add reveal class to key elements
  const revealSelectors = [
    '#sc-1', '#sc-2', '#sc-3', '#sc-4',
    '#dc-1', '#dc-2', '#dc-3', '#dc-4', '#dc-5',
    '#ac-1', '#ac-2', '#ac-3',
    '#hs-1', '#hs-2', '#hs-3', '#hs-4',
    '#ic-1', '#ic-2', '#ic-3',
    '#ms-1', '#ms-2', '#ms-3',
    '.predict-lottie-panel', '.predict-form-card',
    '.mini-timeline', '.newsletter-inner',
  ];
  revealSelectors.forEach((sel, i) => {
    const el = document.querySelector(sel);
    if (!el) return;
    el.classList.add('reveal');
    const delay = (i % 4);
    if (delay) el.classList.add(`reveal-d${delay}`);
    revealObserver.observe(el);
  });

  /* ==========================
     YEAR TIMELINE CHART
     ========================== */
  const yearData = [
    { year: '2015', trips: 5800 },
    { year: '2016', trips: 11400 },
    { year: '2017', trips: 14900 },
    { year: '2018', trips: 18600 },
    { year: '2019', trips: 21300 },
    { year: '2020', trips: 7100 },   // COVID dip
    { year: '2021', trips: 12800 },
    { year: '2022', trips: 19200 },
    { year: '2023', trips: 24700 },
    { year: '2024', trips: 27500 },
    { year: '2025', trips: 30200 },
    { year: '2026', trips: 17800 },  // partial year
  ];
  const maxT = Math.max(...yearData.map(d => d.trips));
  const chartBars = document.getElementById('chart-bars');
  const chartLabels = document.getElementById('chart-labels');
  if (chartBars && chartLabels) {
    yearData.forEach(d => {
      const pct = Math.round((d.trips / maxT) * 100);
      const wrap = document.createElement('div');
      wrap.className = 't-bar-wrap';
      const bar = document.createElement('div');
      bar.className = 't-bar';
      bar.setAttribute('data-val', `~${(d.trips / 1000).toFixed(1)}k`);
      bar.style.height = '0';
      wrap.appendChild(bar);
      chartBars.appendChild(wrap);

      const lbl = document.createElement('div');
      lbl.className = 't-label';
      lbl.textContent = d.year.slice(2); // '15', '16'...
      chartLabels.appendChild(lbl);

      const obs = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting) {
          setTimeout(() => { bar.style.height = pct + '%'; }, 100 + Math.random() * 200);
          obs.unobserve(chartBars);
        }
      }, { threshold: 0.2 });
      obs.observe(chartBars);
    });
  }

  /* ==========================
     ML PREDICTION ENGINE
     ========================== */
  const SEASONAL = {
    1:0.88, 2:0.84, 3:0.92, 4:0.96,
    5:1.02, 6:1.05, 7:1.08, 8:1.10,
    9:1.07, 10:1.04, 11:0.95, 12:0.91
  };
  const BASE_WEIGHT = { small:22, medium:28, large:36, enterprise:48 };

  function predictTrips(month, year, vehicles, shared, baseType) {
    const seasonal = SEASONAL[+month] || 1.0;
    const bw = BASE_WEIGHT[baseType] || 28;
    const v = +vehicles;
    const s = +shared || 0;
    let core = v * bw;
    const sharedBonus = s > 0 ? Math.log1p(s) * 12 : 0;
    let yrFactor = 1.0;
    if (+year >= 2025) yrFactor = 1.12;
    else if (+year >= 2023) yrFactor = 1.05;
    let predicted = core * seasonal * yrFactor + sharedBonus;
    if (v > 100) predicted = predicted * 0.88 + v * 15;
    if (v > 300) predicted = predicted * 0.80 + v * 20;
    predicted = Math.max(10, Math.round(predicted));
    const confidence = Math.min(97, 60 + Math.log10(v + 1) * 18);
    const errPct = baseType === 'enterprise' ? 0.15 : baseType === 'large' ? 0.18 : 0.22;
    return {
      predicted,
      low: Math.round(predicted * (1 - errPct)),
      high: Math.round(predicted * (1 + errPct)),
      confidence: Math.round(confidence)
    };
  }

  function buildInsights(vehicles, month, baseType, predicted) {
    const items = [];
    const m = +month;
    if (m >= 6 && m <= 9) items.push('☀️ Summer season — historically +8–12% above annual average.');
    else if (m === 12 || m === 1) items.push('❄️ Winter season — expect slower growth; plan vehicle readiness.');
    if (+vehicles > 100) items.push('🏢 Large fleet size is the strongest predictor of high dispatch volume.');
    if (baseType === 'enterprise') items.push('⚡ Enterprise bases may show saturation — actual trips might be slightly lower.');
    if (predicted > 20000) items.push('📈 High-volume base. Cross-reference with active license status.');
    if (predicted < 200) items.push('⚠️ Very low volume predicted. Verify vehicle count entry.');
    return items;
  }

  /* ==========================
     FORM HANDLER
     ========================== */
  const predictForm = document.getElementById('predict-form');
  const pfResult = document.getElementById('pf-result');
  const pfResultNum = document.getElementById('pf-result-num');
  const pfLow = document.getElementById('pf-low');
  const pfHigh = document.getElementById('pf-high');
  const pfConfBar = document.getElementById('pf-conf-bar');
  const pfConfLabel = document.getElementById('pf-conf-label');
  const pfInsights = document.getElementById('pf-insights');
  const predictBtn = document.getElementById('predict-btn');

  if (predictForm) {
    predictForm.addEventListener('submit', (e) => {
      e.preventDefault();
      // Clear errors
      predictForm.querySelectorAll('.pf-group.error').forEach(g => g.classList.remove('error'));

      const month    = document.getElementById('pf-month')?.value;
      const year     = document.getElementById('pf-year')?.value;
      const vehicles = document.getElementById('pf-vehicles')?.value;
      const shared   = document.getElementById('pf-shared')?.value;
      const baseType = document.getElementById('pf-base')?.value;

      let valid = true;
      if (!month)   { document.getElementById('pfg-month')?.classList.add('error');   valid = false; }
      if (!year)    { document.getElementById('pfg-year')?.classList.add('error');    valid = false; }
      if (!vehicles || +vehicles < 1) { document.getElementById('pfg-vehicles')?.classList.add('error'); valid = false; }
      if (!baseType){ document.getElementById('pfg-base')?.classList.add('error');    valid = false; }
      if (!valid) return;

      // Show loading
      predictBtn.querySelector('.btn-text').style.display = 'none';
      predictBtn.querySelector('.btn-spin').style.display = 'flex';
      predictBtn.disabled = true;
      if (pfResult) pfResult.style.display = 'none';

      setTimeout(() => {
        const r = predictTrips(month, year, vehicles, shared, baseType);
        const insights = buildInsights(vehicles, month, baseType, r.predicted);

        if (pfResultNum) pfResultNum.textContent = r.predicted.toLocaleString();
        if (pfLow) pfLow.textContent = r.low.toLocaleString();
        if (pfHigh) pfHigh.textContent = r.high.toLocaleString();
        if (pfConfBar) { pfConfBar.style.width = '0%'; setTimeout(() => { pfConfBar.style.width = r.confidence + '%'; }, 60); }
        if (pfConfLabel) pfConfLabel.textContent = `${r.confidence}% Confidence`;
        if (pfInsights) pfInsights.innerHTML = insights.map(i => `<div class="pf-insight-item">${i}</div>`).join('');

        if (pfResult) pfResult.style.display = 'block';

        // Restore button
        predictBtn.querySelector('.btn-text').style.display = 'inline';
        predictBtn.querySelector('.btn-spin').style.display = 'none';
        predictBtn.disabled = false;

        if (window.innerWidth < 900) pfResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 1000);
    });

    // Remove error styling on change
    predictForm.querySelectorAll('input, select').forEach(el => {
      el.addEventListener('change', () => el.closest('.pf-group')?.classList.remove('error'));
      el.addEventListener('input', () => el.closest('.pf-group')?.classList.remove('error'));
    });
  }

  /* ==========================
     NEWSLETTER FORM
     ========================== */
  const nlForm = document.getElementById('newsletter-form');
  if (nlForm) {
    nlForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const emailEl = document.getElementById('newsletter-email');
      const val = emailEl?.value?.trim();
      if (!val || !val.includes('@')) {
        emailEl.style.borderColor = '#EF4444';
        return;
      }
      const btn = document.getElementById('newsletter-submit');
      btn.textContent = '✓ SUBSCRIBED!';
      btn.style.background = '#059669';
      emailEl.value = '';
      setTimeout(() => {
        btn.textContent = 'SUBSCRIBE';
        btn.style.background = '';
      }, 3000);
    });
  }

  /* ==========================
     SEARCH BAR
     ========================== */
  const searchBtn = document.getElementById('search-btn');
  const heroSearch = document.getElementById('hero-search');
  if (searchBtn && heroSearch) {
    searchBtn.addEventListener('click', () => {
      const val = heroSearch.value.trim();
      if (val) {
        console.log('Searching for:', val);
        // Could wire to a real API or filter function
      }
    });
    heroSearch.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') searchBtn.click();
    });
  }

  /* ==========================
     MINI CARD HOVER PARALLAX
     ========================== */
  document.querySelectorAll('.hero-mini-card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const cx = (e.clientX - rect.left) / rect.width - 0.5;
      const cy = (e.clientY - rect.top) / rect.height - 0.5;
      card.style.transform = `translateY(-6px) rotateX(${cy * -8}deg) rotateY(${cx * 8}deg) scale(1.03)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });

  /* ==========================
     DEAL CARD TILT
     ========================== */
  document.querySelectorAll('.deal-card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const cx = (e.clientX - rect.left) / rect.width - 0.5;
      const cy = (e.clientY - rect.top) / rect.height - 0.5;
      card.style.transform = `translateY(-6px) rotateX(${cy * -5}deg) rotateY(${cx * 5}deg)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });

})();
