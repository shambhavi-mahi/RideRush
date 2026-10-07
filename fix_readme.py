import re

with open("README.md", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("?? Key Features", "?? Key Features")
text = text.replace("# RideRush - NYC FHV Intelligence Platform ??", "# RideRush — NYC FHV Intelligence Platform ??")
text = text.replace("## ?? Tech Stack", "## ?? Tech Stack")
text = text.replace("## ?? Architecture & Structure", "## ?? Architecture & Structure")
text = text.replace("## ?? Run Locally", "## ?? Run Locally")
text = text.replace("## ?? About the Dataset", "## ?? About the Dataset")

screenshots_section = """
## ?? Screenshots

### Model Evaluation Dashboard
*(Featuring Color-coded metrics, Jump-links, MathJax formulas, and Heatmap Confusion Matrices)*
<img src="assets/screenshots/evaluation_dashboard.png" width="800">

### Authentication & Login Portal
*(Featuring Full-screen split layout and AI-generated NYC data illustrations)*
<img src="assets/screenshots/login_page.png" width="800">

---
"""

text = text.replace("## ?? Key Features", screenshots_section + "\n## ?? Key Features")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(text)
