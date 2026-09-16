import os

app_path = r"d:\ML\RideRush\RideRush\app.py"
css_path = r"d:\ML\RideRush\RideRush\static\dashboard.css"

with open(app_path, "r", encoding="utf-8") as f:
    app_code = f.read()

# Replace matplotlib colors for light theme
# bg: #f8fafc, surface: #ffffff, border: #e2e8f0, text: #0f172a, text_dim: #475569
app_code = app_code.replace("'#0a1628'", "'#f8fafc'")
app_code = app_code.replace("'#0d1f3c'", "'#ffffff'")
app_code = app_code.replace("'#1e3a5f'", "'#e2e8f0'")
app_code = app_code.replace("'#94a3b8'", "'#475569'")
app_code = app_code.replace("'#e2e8f0'", "'#0f172a'")
app_code = app_code.replace("color='white'", "color='black'")
app_code = app_code.replace("color='black'", "color='#0f172a'") # for texts

with open(app_path, "w", encoding="utf-8") as f:
    f.write(app_code)

with open(css_path, "r", encoding="utf-8") as f:
    css_code = f.read()

# Replace CSS variables
css_vars = """
:root {
  --bg:          #f8fafc;
  --surface:     #ffffff;
  --surface2:    #f1f5f9;
  --border:      #e2e8f0;
  --orange:      #F59E0B;
  --orange-dim:  rgba(245,158,11,0.15);
  --blue:        #3B82F6;
  --green:       #10B981;
  --red:         #EF4444;
  --purple:      #8B5CF6;
  --text:        #0f172a;
  --text-muted:  #64748b;
  --text-dim:    #475569;
  --sidebar-w:   220px;
  --radius:      12px;
  --radius-sm:   8px;
  --font:        'DM Sans', sans-serif;
  --font-display:'Sora', sans-serif;
}
"""
import re
css_code = re.sub(r":root\s*\{[^}]+\}", css_vars.strip(), css_code)
css_code = css_code.replace("rgba(255,255,255,0.025)", "rgba(0,0,0,0.025)")

with open(css_path, "w", encoding="utf-8") as f:
    f.write(css_code)
