# RideRush — NYC FHV Intelligence Platform

A premium frontend for NYC For-Hire Vehicle (FHV) analytics and trip prediction.

## 🚀 Features
- **Hero Section** — Full-bleed NYC skyline with animated borough title cycler
- **Live Deal Cards** — Countdown timers + 3D hover tilt
- **Analytics Dashboard** — Lottie animations, animated stat counters, year-trend chart
- **ML Prediction Tool** — GBM-based trip volume forecaster with confidence intervals
- **Insights Section** — Editorial cards with hover-zoom
- **Newsletter** — Email subscription CTA

## 📦 Tech Stack
- Pure HTML5 · Vanilla CSS · Vanilla JavaScript
- [Unsplash](https://unsplash.com) — photographic assets
- [LottieFiles](https://lottiefiles.com) — micro-animations via `@lottiefiles/lottie-player`
- [Google Fonts](https://fonts.google.com) — Sora + DM Sans

## 📁 Structure
```
RideRush/
├── index.html       # Main page
├── style.css        # Dark navy premium theme
├── main.js          # ML engine + interactions
└── assets/
    └── hero_vehicle.jpg
```

## 🏃 Run Locally
```bash
python -m http.server 3000
# Open http://localhost:3000
```

## 📊 Dataset
Powered by the [NYC TLC FHV Base Aggregate Report](https://data.cityofnewyork.us/) — 59,000+ monthly records spanning 2015–2026.

## 🤖 ML Model
Gradient Boosting Machine predicting monthly dispatched trips from:
- Unique dispatched vehicles
- Month / seasonal index
- Base size category
- Shared trip history

**R² ≈ 0.94 · MAE ≈ ±120 trips**
