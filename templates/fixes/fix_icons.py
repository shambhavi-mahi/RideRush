import re

with open("d:/ML/RideRush/RideRush/index.html", "r", encoding="utf-8") as f:
    html = f.read()

replacements = {
    "https://assets4.lottiefiles.com/packages/lf20_hm1mkcgk.json": """<svg viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="width:100%;height:100%;"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>""",
    
    "https://assets5.lottiefiles.com/packages/lf20_nk4jvsun.json": """<svg viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:100%;height:100%;"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>""",
    
    "https://assets3.lottiefiles.com/packages/lf20_jmejybvu.json": """<svg viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:100%;height:100%;"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>""",
    
    "https://assets9.lottiefiles.com/packages/lf20_qp1q7mct.json": """<svg viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:100%;height:100%;"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>""",
    
    "https://assets10.lottiefiles.com/packages/lf20_t9gkkhz4.json": """<svg viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:100%;height:100%;"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>""",
    
    "https://assets5.lottiefiles.com/packages/lf20_jdqfpewq.json": """<svg viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="width:100%;height:100%;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>""",
    
    "https://assets3.lottiefiles.com/packages/lf20_fcfjwiyb.json": """<svg viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="width:100%;height:100%;"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>""",
}

def replace_lottie(match):
    tag = match.group(0)
    for url, svg in replacements.items():
        if url in tag:
            return svg
    return tag

html = re.sub(r'<lottie-player[^>]+>.*?</lottie-player>', replace_lottie, html, flags=re.DOTALL)

with open("d:/ML/RideRush/RideRush/index.html", "w", encoding="utf-8") as f:
    f.write(html)
