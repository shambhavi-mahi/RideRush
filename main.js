/* ===================================================
   RideRush — main.js
   Three.js particle network + GSAP animations +
   ML prediction engine + UI interactions
   =================================================== */

(function () {
  'use strict';

  /* =====================================================
     1. THREE.JS — Hero Particle Network
     ===================================================== */
  (function initThree() {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas || typeof THREE === 'undefined') return;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);

    // Scene & Camera
    const scene = new THREE.Scene();
    const W = canvas.clientWidth  || window.innerWidth;
    const H = canvas.clientHeight || window.innerHeight;
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000);
    camera.position.z = 90;

    renderer.setSize(W, H);

    // ---- Particles ----
    const COUNT       = 180;
    const SPREAD_X    = 140;
    const SPREAD_Y    = 90;
    const SPREAD_Z    = 60;
    const MAX_DIST    = 28;        // max distance for line connection
    const MAX_LINES   = 500;

    const positions  = new Float32Array(COUNT * 3);
    const velocities = [];
    const sizes      = new Float32Array(COUNT);

    for (let i = 0; i < COUNT; i++) {
      positions[i * 3]     = (Math.random() - 0.5) * SPREAD_X;
      positions[i * 3 + 1] = (Math.random() - 0.5) * SPREAD_Y;
      positions[i * 3 + 2] = (Math.random() - 0.5) * SPREAD_Z;
      velocities.push({
        x: (Math.random() - 0.5) * 0.018,
        y: (Math.random() - 0.5) * 0.012,
        z: (Math.random() - 0.5) * 0.008,
      });
      sizes[i] = Math.random() * 1.5 + 0.5;
    }

    const ptGeo = new THREE.BufferGeometry();
    ptGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    ptGeo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const ptMat = new THREE.PointsMaterial({
      color: 0xF59E0B,
      size: 1.2,
      transparent: true,
      opacity: 0.75,
      sizeAttenuation: true,
    });

    const points = new THREE.Points(ptGeo, ptMat);
    scene.add(points);

    // ---- Connection Lines ----
    const linePositions = new Float32Array(MAX_LINES * 2 * 3);
    const lineOpacities = new Float32Array(MAX_LINES * 2);

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    lineGeo.setDrawRange(0, 0);

    const lineMat = new THREE.LineBasicMaterial({
      color: 0xF59E0B,
      transparent: true,
      opacity: 0.18,
    });

    const lineSegments = new THREE.LineSegments(lineGeo, lineMat);
    scene.add(lineSegments);

    // ---- Secondary smaller white particles ----
    const COUNT2 = 60;
    const pos2   = new Float32Array(COUNT2 * 3);
    for (let i = 0; i < COUNT2; i++) {
      pos2[i * 3]     = (Math.random() - 0.5) * SPREAD_X * 1.3;
      pos2[i * 3 + 1] = (Math.random() - 0.5) * SPREAD_Y * 1.3;
      pos2[i * 3 + 2] = (Math.random() - 0.5) * SPREAD_Z;
    }
    const ptGeo2 = new THREE.BufferGeometry();
    ptGeo2.setAttribute('position', new THREE.BufferAttribute(pos2, 3));
    const ptMat2 = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.4,
      transparent: true,
      opacity: 0.3,
      sizeAttenuation: true,
    });
    scene.add(new THREE.Points(ptGeo2, ptMat2));

    // ---- Mouse Tracking ----
    let targetX = 0, targetY = 0;
    let camX = 0, camY = 0;
    window.addEventListener('mousemove', (e) => {
      targetX = (e.clientX / window.innerWidth  - 0.5) * 14;
      targetY = -(e.clientY / window.innerHeight - 0.5) * 8;
    }, { passive: true });

    // ---- Resize ----
    window.addEventListener('resize', () => {
      const w = canvas.clientWidth  || window.innerWidth;
      const h = canvas.clientHeight || window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }, { passive: true });

    // ---- Clock for time-based effects ----
    const clock = new THREE.Clock();

    // ---- Animation Loop ----
    function animate() {
      requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      // Smooth camera parallax following mouse
      camX += (targetX - camX) * 0.025;
      camY += (targetY - camY) * 0.025;
      camera.position.x = camX;
      camera.position.y = camY;
      camera.lookAt(scene.position);

      // Update particle positions (drift + soft boundary wrap)
      for (let i = 0; i < COUNT; i++) {
        const ix = i * 3, iy = i * 3 + 1, iz = i * 3 + 2;
        positions[ix] += velocities[i].x;
        positions[iy] += velocities[i].y;
        positions[iz] += velocities[i].z;
        // Wrap at boundaries
        if (positions[ix]  >  SPREAD_X / 2) positions[ix]  = -SPREAD_X / 2;
        if (positions[ix]  < -SPREAD_X / 2) positions[ix]  =  SPREAD_X / 2;
        if (positions[iy]  >  SPREAD_Y / 2) positions[iy]  = -SPREAD_Y / 2;
        if (positions[iy]  < -SPREAD_Y / 2) positions[iy]  =  SPREAD_Y / 2;
        if (positions[iz]  >  SPREAD_Z / 2) positions[iz]  = -SPREAD_Z / 2;
        if (positions[iz]  < -SPREAD_Z / 2) positions[iz]  =  SPREAD_Z / 2;
      }
      ptGeo.attributes.position.needsUpdate = true;

      // Pulse opacity of points
      ptMat.opacity = 0.55 + Math.sin(elapsed * 0.8) * 0.2;

      // Slow rotation of whole particle field
      points.rotation.y = elapsed * 0.04;
      points.rotation.x = elapsed * 0.015;

      // Rebuild line connections
      let lineCount = 0;
      const lp = lineGeo.attributes.position.array;

      for (let a = 0; a < COUNT && lineCount < MAX_LINES; a++) {
        const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
        for (let b = a + 1; b < COUNT && lineCount < MAX_LINES; b++) {
          const dx = ax - positions[b * 3];
          const dy = ay - positions[b * 3 + 1];
          const dz = az - positions[b * 3 + 2];
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (dist < MAX_DIST) {
            const idx = lineCount * 6;
            lp[idx]     = ax;  lp[idx + 1] = ay;  lp[idx + 2] = az;
            lp[idx + 3] = positions[b * 3]; lp[idx + 4] = positions[b * 3 + 1]; lp[idx + 5] = positions[b * 3 + 2];
            lineCount++;
          }
        }
      }
      lineGeo.attributes.position.needsUpdate = true;
      lineGeo.setDrawRange(0, lineCount * 2);
      lineMat.opacity = 0.10 + Math.sin(elapsed * 0.5) * 0.06;

      renderer.render(scene, camera);
    }

    animate();
  })();


  /* =====================================================
     2. GSAP — Hero Entrance Timeline
     ===================================================== */
  (function initHeroGSAP() {
    if (typeof gsap === 'undefined') return;

    gsap.registerPlugin(ScrollTrigger);

    // Master hero timeline — each element flies in sequentially
    const tl = gsap.timeline({ delay: 0.25, defaults: { ease: 'power3.out' } });

    tl
      .to('.hero-location-chip', {
        opacity: 1, y: 0, duration: 0.7,
        from: { opacity: 0, y: 30 }
      })
      .fromTo('.hero-title',
        { opacity: 0, y: 60, skewY: 3 },
        { opacity: 1, y: 0, skewY: 0, duration: 1.0 },
        '-=0.3'
      )
      .fromTo('.hero-subtitle',
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.65 },
        '-=0.55'
      )
      .fromTo('.hero-meta',
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.6 },
        '-=0.45'
      )
      .fromTo('.hero-selectors',
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.6 },
        '-=0.45'
      )
      .fromTo('.hero-search-wrap',
        { opacity: 0, y: 20, scale: 0.97 },
        { opacity: 1, y: 0, scale: 1, duration: 0.65 },
        '-=0.4'
      )
      .fromTo('#hmc-1',
        { opacity: 0, x: 50, rotation: 4 },
        { opacity: 1, x: 0, rotation: 0, duration: 0.75 },
        '-=0.5'
      )
      .fromTo('#hmc-2',
        { opacity: 0, x: 50, rotation: 4 },
        { opacity: 1, x: 0, rotation: 0, duration: 0.75 },
        '-=0.55'
      )
      .fromTo('.scroll-indicator',
        { opacity: 0 },
        { opacity: 1, duration: 0.5 },
        '-=0.3'
      );

    // Navbar letters pop in
    gsap.fromTo('.nav-logo',
      { opacity: 0, x: -16 },
      { opacity: 1, x: 0, duration: 0.7, ease: 'power2.out', delay: 0.1 }
    );
    gsap.fromTo('.nav-links li',
      { opacity: 0, y: -12 },
      { opacity: 1, y: 0, duration: 0.5, stagger: 0.08, ease: 'power2.out', delay: 0.3 }
    );
    gsap.fromTo('#nav-cta',
      { opacity: 0, scale: 0.85 },
      { opacity: 1, scale: 1, duration: 0.5, ease: 'back.out(1.7)', delay: 0.6 }
    );


    /* =====================================================
       3. GSAP ScrollTrigger — Section reveals
       ===================================================== */

    // --- Deals Section: cards stagger slide up ---
    gsap.fromTo('.deals-title',
      { opacity: 0, x: -40 },
      {
        opacity: 1, x: 0, duration: 0.8, ease: 'power3.out',
        scrollTrigger: { trigger: '.deals-section', start: 'top 80%' }
      }
    );
    gsap.fromTo('.btn-outline-orange',
      { opacity: 0, x: 40 },
      {
        opacity: 1, x: 0, duration: 0.7, ease: 'power3.out',
        scrollTrigger: { trigger: '.deals-section', start: 'top 80%' }
      }
    );
    gsap.fromTo('.deal-card',
      { opacity: 0, y: 70, scale: 0.93 },
      {
        opacity: 1, y: 0, scale: 1,
        duration: 0.75, stagger: 0.1, ease: 'power3.out',
        scrollTrigger: { trigger: '.deals-track', start: 'top 85%' }
      }
    );

    // --- Analytics Section ---
    gsap.fromTo('.analytics-left',
      { opacity: 0, x: -60 },
      {
        opacity: 1, x: 0, duration: 0.9, ease: 'power3.out',
        scrollTrigger: { trigger: '.analytics-section', start: 'top 75%' }
      }
    );
    gsap.fromTo('.stat-card',
      { opacity: 0, scale: 0.8, y: 30 },
      {
        opacity: 1, scale: 1, y: 0,
        duration: 0.65, stagger: 0.12, ease: 'back.out(1.5)',
        scrollTrigger: { trigger: '.stat-grid', start: 'top 82%' }
      }
    );
    gsap.fromTo('.mini-timeline',
      { opacity: 0, y: 40 },
      {
        opacity: 1, y: 0, duration: 0.8, ease: 'power2.out',
        scrollTrigger: { trigger: '.mini-timeline', start: 'top 85%' }
      }
    );

    // --- Newsletter ---
    gsap.fromTo('.newsletter-title',
      { opacity: 0, y: 30 },
      {
        opacity: 1, y: 0, duration: 0.8, ease: 'power3.out',
        scrollTrigger: { trigger: '.newsletter-section', start: 'top 80%' }
      }
    );
    gsap.fromTo('.newsletter-form',
      { opacity: 0, scale: 0.95, y: 20 },
      {
        opacity: 1, scale: 1, y: 0, duration: 0.7, ease: 'power2.out',
        scrollTrigger: { trigger: '.newsletter-section', start: 'top 80%' }
      }
    );

    // --- Prediction Section ---
    gsap.fromTo('.predict-lottie-panel',
      { opacity: 0, x: -50 },
      {
        opacity: 1, x: 0, duration: 0.85, ease: 'power3.out',
        scrollTrigger: { trigger: '.predict-section', start: 'top 75%' }
      }
    );
    gsap.fromTo('.predict-form-card',
      { opacity: 0, x: 50 },
      {
        opacity: 1, x: 0, duration: 0.85, ease: 'power3.out',
        scrollTrigger: { trigger: '.predict-section', start: 'top 75%' }
      }
    );

    // --- Insights ---
    gsap.fromTo('.insights-title-row',
      { opacity: 0, y: 30 },
      {
        opacity: 1, y: 0, duration: 0.7, ease: 'power3.out',
        scrollTrigger: { trigger: '.insights-section', start: 'top 80%' }
      }
    );
    gsap.fromTo('.insight-card',
      { opacity: 0, y: 60, scale: 0.95 },
      {
        opacity: 1, y: 0, scale: 1,
        duration: 0.75, stagger: 0.15, ease: 'power3.out',
        scrollTrigger: { trigger: '.insights-grid', start: 'top 82%' }
      }
    );

    // --- How It Works ---
    gsap.fromTo('.how-step',
      { opacity: 0, y: 50 },
      {
        opacity: 1, y: 0,
        duration: 0.7, stagger: 0.15, ease: 'power3.out',
        scrollTrigger: { trigger: '.how-section', start: 'top 78%' }
      }
    );

    // --- Footer ---
    gsap.fromTo('.footer-brand',
      { opacity: 0, y: 30 },
      {
        opacity: 1, y: 0, duration: 0.7, ease: 'power2.out',
        scrollTrigger: { trigger: '.footer', start: 'top 90%' }
      }
    );
    gsap.fromTo('.footer-col',
      { opacity: 0, y: 20 },
      {
        opacity: 1, y: 0, duration: 0.6, stagger: 0.1, ease: 'power2.out',
        scrollTrigger: { trigger: '.footer', start: 'top 90%' }
      }
    );

    // --- Parallax: hero bg image scrolls slower ---
    gsap.to('.hero-bg-img', {
      yPercent: 30,
      ease: 'none',
      scrollTrigger: {
        trigger: '.hero',
        start: 'top top',
        end: 'bottom top',
        scrub: true,
      }
    });

    // --- Number counters triggered by ScrollTrigger ---
    document.querySelectorAll('[data-count]').forEach(el => {
      const target = parseInt(el.getAttribute('data-count'), 10);
      gsap.fromTo({ val: 0 }, { val: target,
        duration: 2.2,
        ease: 'power2.out',
        onUpdate: function () { el.textContent = Math.round(this.targets()[0].val).toLocaleString(); },
        scrollTrigger: { trigger: el, start: 'top 85%', once: true }
      });
    });

  })();


  /* =====================================================
     4. NAVBAR scroll behavior
     ===================================================== */
  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 40);
  }, { passive: true });


  /* =====================================================
     5. HAMBURGER
     ===================================================== */
  const hamburger = document.getElementById('hamburger');
  const mobileMenu = document.getElementById('mobile-menu');
  if (hamburger) {
    hamburger.addEventListener('click', () => mobileMenu.classList.toggle('open'));
    mobileMenu.querySelectorAll('a').forEach(a =>
      a.addEventListener('click', () => mobileMenu.classList.remove('open'))
    );
  }


  /* =====================================================
     6. HERO BG image zoom on load
     ===================================================== */
  const heroBgImg = document.getElementById('hero-bg-img');
  if (heroBgImg) {
    if (heroBgImg.complete) heroBgImg.classList.add('loaded');
    else heroBgImg.addEventListener('load', () => heroBgImg.classList.add('loaded'));
  }


  /* =====================================================
     7. HERO TITLE CYCLER (borough names)
     ===================================================== */
  const boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'The Bronx', 'Staten Island'];
  let titleIdx = 0;
  const heroTitle = document.getElementById('hero-title');
  if (heroTitle && typeof gsap !== 'undefined') {
    setInterval(() => {
      gsap.to(heroTitle, {
        opacity: 0, y: 16, skewY: 2, duration: 0.35, ease: 'power2.in',
        onComplete: () => {
          titleIdx = (titleIdx + 1) % boroughs.length;
          heroTitle.textContent = boroughs[titleIdx];
          gsap.to(heroTitle, { opacity: 1, y: 0, skewY: 0, duration: 0.5, ease: 'power3.out' });
        }
      });
    }, 3200);
  }


  /* =====================================================
     8. COUNTDOWN TIMERS on deal cards
     ===================================================== */
  document.querySelectorAll('.timer-val').forEach(el => {
    let secs = parseInt(el.getAttribute('data-seconds'), 10);
    const tick = () => {
      if (secs <= 0) { el.textContent = '00:00:00'; return; }
      secs--;
      el.textContent = [
        Math.floor(secs / 3600),
        Math.floor((secs % 3600) / 60),
        secs % 60,
      ].map(n => n.toString().padStart(2, '0')).join(':');
    };
    tick();
    setInterval(tick, 1000);
  });


  /* =====================================================
     9. DEALS TRACK SCROLL ARROWS
     ===================================================== */
  const track   = document.getElementById('deals-track');
  const prevBtn = document.getElementById('track-prev');
  const nextBtn = document.getElementById('track-next');
  if (track && prevBtn && nextBtn) {
    const SCROLL = 280;
    nextBtn.addEventListener('click', () => track.scrollBy({ left:  SCROLL, behavior: 'smooth' }));
    prevBtn.addEventListener('click', () => track.scrollBy({ left: -SCROLL, behavior: 'smooth' }));
  }


  /* =====================================================
     10. YEAR TIMELINE CHART (analytics section)
     ===================================================== */
  const yearData = [
    { y: '15', v: 5800  }, { y: '16', v: 11400 }, { y: '17', v: 14900 },
    { y: '18', v: 18600 }, { y: '19', v: 21300 }, { y: '20', v: 7100  },
    { y: '21', v: 12800 }, { y: '22', v: 19200 }, { y: '23', v: 24700 },
    { y: '24', v: 27500 }, { y: '25', v: 30200 }, { y: '26', v: 17800 },
  ];
  const maxV = Math.max(...yearData.map(d => d.v));
  const chartBars   = document.getElementById('chart-bars');
  const chartLabels = document.getElementById('chart-labels');

  if (chartBars && chartLabels) {
    yearData.forEach((d, i) => {
      const pct  = Math.round((d.v / maxV) * 100);
      const wrap = document.createElement('div');
      wrap.className = 't-bar-wrap';
      const bar = document.createElement('div');
      bar.className = 't-bar';
      bar.setAttribute('data-val', `~${(d.v / 1000).toFixed(1)}k`);
      bar.style.height = '0';
      wrap.appendChild(bar);
      chartBars.appendChild(wrap);

      const lbl = document.createElement('div');
      lbl.className = 't-label';
      lbl.textContent = d.y;
      chartLabels.appendChild(lbl);

      // Animate bar height via GSAP ScrollTrigger if available, else IntersectionObserver
      if (typeof gsap !== 'undefined') {
        gsap.to(bar, {
          height: pct + '%',
          duration: 1.2, delay: i * 0.06,
          ease: 'power3.out',
          scrollTrigger: { trigger: chartBars, start: 'top 85%', once: true }
        });
      } else {
        const obs = new IntersectionObserver((entries) => {
          if (entries[0].isIntersecting) {
            setTimeout(() => { bar.style.height = pct + '%'; }, 80 + i * 60);
            obs.disconnect();
          }
        }, { threshold: 0.2 });
        obs.observe(chartBars);
      }
    });
  }


  /* =====================================================
     11. ML PREDICTION ENGINE (Now hitting Flask backend)
     ===================================================== */


  function buildInsights(vehicles, month, baseType, predicted) {
    const items = [];
    const m = +month;
    if (m >= 6 && m <= 9) items.push('☀️ Summer season — historically +8–12% above annual average.');
    else if (m === 12 || m === 1) items.push('❄️ Winter season — expect slower growth; plan vehicle readiness.');
    if (+vehicles > 100) items.push('🏢 Large fleet size is the strongest predictor of high dispatch volume.');
    if (baseType === 'enterprise') items.push('⚡ Enterprise bases may show saturation — actual trips might be slightly lower.');
    if (predicted > 20000) items.push('📈 High-volume base. Cross-reference with active license status.');
    if (predicted < 200)  items.push('⚠️ Very low volume predicted. Double-check vehicle count entry.');
    return items;
  }

  const predictForm  = document.getElementById('predict-form');
  const pfResult     = document.getElementById('pf-result');
  const pfResultNum  = document.getElementById('pf-result-num');
  const pfLow        = document.getElementById('pf-low');
  const pfHigh       = document.getElementById('pf-high');
  const pfConfBar    = document.getElementById('pf-conf-bar');
  const pfConfLabel  = document.getElementById('pf-conf-label');
  const pfInsights   = document.getElementById('pf-insights');
  const predictBtn   = document.getElementById('predict-btn');

  if (predictForm) {
    predictForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      predictForm.querySelectorAll('.pf-group.error').forEach(g => g.classList.remove('error'));

      const month    = document.getElementById('pf-month')?.value;
      const year     = document.getElementById('pf-year')?.value;
      const vehicles = document.getElementById('pf-vehicles')?.value;
      const shared   = document.getElementById('pf-shared')?.value || 0;
      const baseType = document.getElementById('pf-base')?.value;

      let valid = true;
      if (!month)                    { document.getElementById('pfg-month')?.classList.add('error');    valid = false; }
      if (!year)                     { document.getElementById('pfg-year')?.classList.add('error');     valid = false; }
      if (!vehicles || +vehicles<1) { document.getElementById('pfg-vehicles')?.classList.add('error'); valid = false; }
      if (!baseType)                 { document.getElementById('pfg-base')?.classList.add('error');     valid = false; }
      if (!valid) return;

      predictBtn.querySelector('.btn-text').style.display = 'none';
      predictBtn.querySelector('.btn-spin').style.display = 'flex';
      predictBtn.disabled = true;
      if (pfResult) pfResult.style.display = 'none';

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ month, year, vehicles, shared, baseType })
        });
        
        const r = await response.json();
        
        if (response.ok) {
          const insights = buildInsights(vehicles, month, baseType, r.predicted);

          if (pfResultNum) {
            pfResultNum.textContent = '0';
            // Animate the result number with GSAP
            if (typeof gsap !== 'undefined') {
              gsap.fromTo({ val: 0 }, { val: r.predicted,
                duration: 1.4, ease: 'power3.out',
                onUpdate: function () { pfResultNum.textContent = Math.round(this.targets()[0].val).toLocaleString(); }
              });
              // Bounce-in the result card
              if (pfResult) {
                pfResult.style.display = 'block';
                gsap.fromTo(pfResult, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.5, ease: 'power2.out' });
              }
            } else {
              pfResultNum.textContent = r.predicted.toLocaleString();
              if (pfResult) pfResult.style.display = 'block';
            }
          }

          if (pfLow)        pfLow.textContent  = r.low.toLocaleString();
          if (pfHigh)       pfHigh.textContent = r.high.toLocaleString();
          if (pfConfBar) {
            pfConfBar.style.width = '0%';
            setTimeout(() => { pfConfBar.style.width = r.confidence + '%'; }, 60);
          }
          if (pfConfLabel)  pfConfLabel.textContent = `${r.confidence}% Confidence`;
          if (pfInsights)   pfInsights.innerHTML = insights.map(i => `<div class="pf-insight-item">${i}</div>`).join('');
          
          if (window.innerWidth < 900 && pfResult) pfResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
          console.error("Backend error:", r.error);
          alert("Error from prediction server: " + (r.error || "Unknown"));
        }
      } catch (err) {
        console.error("Fetch failed:", err);
        alert("Failed to connect to the prediction backend. Is app.py running?");
      } finally {
        predictBtn.querySelector('.btn-text').style.display = 'inline';
        predictBtn.querySelector('.btn-spin').style.display = 'none';
        predictBtn.disabled = false;
      }
    });

    predictForm.querySelectorAll('input, select').forEach(el => {
      el.addEventListener('change', () => el.closest('.pf-group')?.classList.remove('error'));
      el.addEventListener('input',  () => el.closest('.pf-group')?.classList.remove('error'));
    });
  }


  /* =====================================================
     12. NEWSLETTER FORM
     ===================================================== */
  const nlForm = document.getElementById('newsletter-form');
  if (nlForm) {
    nlForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const emailEl = document.getElementById('newsletter-email');
      if (!emailEl.value.trim().includes('@')) {
        if (typeof gsap !== 'undefined') {
          gsap.fromTo(emailEl, { x: 0 }, { x: [-8, 8, -6, 6, -4, 0], duration: 0.4, ease: 'power1.inOut' });
        }
        emailEl.style.borderColor = '#EF4444';
        return;
      }
      const btn = document.getElementById('newsletter-submit');
      btn.textContent = '✓ SUBSCRIBED!';
      btn.style.background = '#059669';
      emailEl.value = '';
      setTimeout(() => { btn.textContent = 'SUBSCRIBE'; btn.style.background = ''; }, 3000);
    });
  }


  /* =====================================================
     13. SEARCH BAR
     ===================================================== */
  const searchBtn  = document.getElementById('search-btn');
  const heroSearch = document.getElementById('hero-search');
  if (searchBtn && heroSearch) {
    searchBtn.addEventListener('click', () => console.log('Search:', heroSearch.value.trim()));
    heroSearch.addEventListener('keydown', (e) => { if (e.key === 'Enter') searchBtn.click(); });
  }


  /* =====================================================
     14. CARD 3D TILT (GSAP QuickSetter)
     ===================================================== */
  function addTilt(selector) {
    document.querySelectorAll(selector).forEach(card => {
      card.addEventListener('mousemove', (e) => {
        const r = card.getBoundingClientRect();
        const cx = (e.clientX - r.left) / r.width  - 0.5;
        const cy = (e.clientY - r.top)  / r.height - 0.5;
        if (typeof gsap !== 'undefined') {
          gsap.to(card, {
            rotateX: cy * -9, rotateY: cx * 9,
            transformPerspective: 900,
            translateY: -6,
            duration: 0.35, ease: 'power2.out',
            overwrite: 'auto',
          });
        }
      });
      card.addEventListener('mouseleave', () => {
        if (typeof gsap !== 'undefined') {
          gsap.to(card, { rotateX: 0, rotateY: 0, translateY: 0, duration: 0.55, ease: 'power3.out' });
        }
      });
    });
  }
  addTilt('.deal-card');
  addTilt('.hero-mini-card');
  addTilt('.stat-card');

})();
