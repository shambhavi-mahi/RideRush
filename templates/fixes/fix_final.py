import os

# 1. Fix style.css
with open("d:/ML/RideRush/RideRush/style.css", "rb") as f:
    content = f.read()

# Try to decode as utf-8, but it might have the utf-16 junk at the end
try:
    text = content.decode('utf-8', errors='ignore')
except:
    text = content.decode('utf-16le', errors='ignore')

# Find the start of the junk or the media query end
idx = text.rfind("}\n\n / *")
if idx == -1:
    idx = text.rfind("}\n\n\x00 ")

if idx != -1:
    text = text[:idx + 2] # Keep the closing bracket of the media query and newline

# Append correct, smaller CSS sizes using standard !important to override inline styles
correct_css = """

/* --- SVG Icon Size Fixes --- */
.analytics-lottie svg { width: 180px !important; height: 180px !important; }
.stat-card-icon svg { width: 36px !important; height: 36px !important; }
.newsletter-lottie svg { width: 48px !important; height: 48px !important; }
.predict-lottie-panel svg { width: 180px !important; height: 180px !important; margin: 0 auto; display: block; }
.play-btn svg { width: 40px !important; height: 40px !important; margin: 0 auto; display: block; }
.btn-spin svg { width: 18px !important; height: 18px !important; }
.hs-icon svg { width: 48px !important; height: 48px !important; }
"""

with open("d:/ML/RideRush/RideRush/style.css", "w", encoding="utf-8") as f:
    f.write(text.strip() + correct_css)

# 2. Fix index.html (change newsletter play button to mail icon)
with open("d:/ML/RideRush/RideRush/index.html", "r", encoding="utf-8") as f:
    html = f.read()

bad_newsletter = '<div class="newsletter-lottie">\n        <svg viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="width:100%;height:100%;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>\n      </div>'
good_newsletter = '<div class="newsletter-lottie">\n        <svg viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="width:100%;height:100%;"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>\n      </div>'

html = html.replace(bad_newsletter, good_newsletter)

with open("d:/ML/RideRush/RideRush/index.html", "w", encoding="utf-8") as f:
    f.write(html)
