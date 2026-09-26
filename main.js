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

    // Allow mouse interaction on canvas
    canvas.style.pointerEvents = 'auto';

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);

    const scene = new THREE.Scene();
    const W = canvas.clientWidth  || window.innerWidth;
    const H = canvas.clientHeight || window.innerHeight;
    const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 1000);
    camera.position.set(0, 5, 110);

    renderer.setSize(W, H);

    // ---- Lighting ----
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
    dirLight.position.set(60, 80, 60);
    dirLight.castShadow = true;
    scene.add(dirLight);
    const fillLight = new THREE.DirectionalLight(0xF59E0B, 0.3);
    fillLight.position.set(-40, -20, 30);
    scene.add(fillLight);

    // ---- Solid Golden Torus Knot (lower-left) ----
    const torusGeom = new THREE.TorusKnotGeometry(16, 7.5, 256, 64, 2, 3);
    const torusMat  = new THREE.MeshPhysicalMaterial({
      color: 0xF59E0B,
      roughness: 0.12,
      metalness: 0.08,
      clearcoat: 1.0,
      clearcoatRoughness: 0.15,
    });
    const torus = new THREE.Mesh(torusGeom, torusMat);
    torus.position.set(-42, -12, 15);
    torus.castShadow = true;
    torus.receiveShadow = true;
    scene.add(torus);

    // ---- Particle Wave Field ----
    const COUNT = 4500;
    const baseX = new Float32Array(COUNT);   // store original X for color mapping
    const pos   = new Float32Array(COUNT * 3);
    const clrs  = new Float32Array(COUNT * 3);

    const colOrange = new THREE.Color(0xF59E0B);
    const colNavy   = new THREE.Color(0x1e3a5f);

    for (let i = 0; i < COUNT; i++) {
      // Spread particles across a wide horizontal band, biased right
      const x = (Math.random() - 0.35) * 320;
      const z = (Math.random() - 0.5) * 100;
      baseX[i]        = x;
      pos[i * 3]      = x;
      pos[i * 3 + 1]  = 0;
      pos[i * 3 + 2]  = z;

      // Color gradient: orange on far-left → blended middle → dark navy on far-right
      const t = Math.min(1, Math.max(0, (x + 120) / 280));
      const noisy = Math.min(1, Math.max(0, t + (Math.random() - 0.5) * 0.15));
      const c = new THREE.Color().lerpColors(colOrange, colNavy, noisy);
      clrs[i * 3]     = c.r;
      clrs[i * 3 + 1] = c.g;
      clrs[i * 3 + 2] = c.b;
    }

    const ptGeo = new THREE.BufferGeometry();
    ptGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    ptGeo.setAttribute('color',    new THREE.BufferAttribute(clrs, 3));

    const ptMat = new THREE.PointsMaterial({
      size: 1.6,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true,
    });
    const points = new THREE.Points(ptGeo, ptMat);
    points.position.set(10, -5, 0);   // shift wave band slightly right & down
    scene.add(points);

    // ---- Mouse Parallax ----
    let mx = 0, my = 0, camX = 0, camY = 0;
    window.addEventListener('mousemove', (e) => {
      mx = (e.clientX / window.innerWidth  - 0.5) * 12;
      my = -(e.clientY / window.innerHeight - 0.5) * 6;
    }, { passive: true });

    // ---- Resize ----
    window.addEventListener('resize', () => {
      const w = canvas.clientWidth  || window.innerWidth;
      const h = canvas.clientHeight || window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }, { passive: true });

    // ---- Render Loop ----
    const clock = new THREE.Clock();

    function animate() {
      requestAnimationFrame(animate);
      const t = clock.getElapsedTime();

      // Smooth camera follow
      camX += (mx - camX) * 0.04;
      camY += (my - camY) * 0.04;
      camera.position.x = camX;
      camera.position.y = 5 + camY;
      camera.lookAt(0, 0, 0);

      // Torus rotation
      torus.rotation.x = t * 0.18;
      torus.rotation.y = t * 0.25;

      // Animate wave
      const pArr = ptGeo.attributes.position.array;
      for (let i = 0; i < COUNT; i++) {
        const x = baseX[i];
        const z = pArr[i * 3 + 2];
        const w1 = Math.sin(x * 0.022 + t * 0.7) * 18;
        const w2 = Math.cos(x * 0.014 - t * 0.45 + z * 0.012) * 12;
        const w3 = Math.sin(z * 0.028 + t * 0.9) * 6;
        pArr[i * 3 + 1] = w1 + w2 + w3;
      }
      ptGeo.attributes.position.needsUpdate = true;

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
      const proxy  = { val: 0 };
      gsap.to(proxy, {
        val: target,
        duration: 2.2,
        ease: 'power2.out',
        onUpdate: function () {
          el.textContent = Math.round(proxy.val).toLocaleString();
        },
        scrollTrigger: { trigger: el, start: 'top 90%', once: true }
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
  const heroBgImg = document.getElementById('hero-bg-img') || document.querySelector('.hero-bg-light');
  if (heroBgImg) {
    if (heroBgImg.complete) heroBgImg.classList.add('loaded');
    else heroBgImg.addEventListener('load', () => heroBgImg.classList.add('loaded'));
  }


  /* =====================================================
     7. HERO TITLE CYCLER (borough names)
     ===================================================== */
  const boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'The Bronx', 'Staten Island'];
  let titleIdx = 0;
  const heroTitle = document.querySelector('.hc-title');
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
  const searchBtn  = document.querySelector('.hc-search-btn');
  const heroSearch = document.querySelector('.hc-search-input');
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
