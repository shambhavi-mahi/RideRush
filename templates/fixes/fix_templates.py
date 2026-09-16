import os

template_dir = r"d:\ML\RideRush\RideRush\templates"

replacements = {
    '#e2e8f0': '#0f172a',
    '#94a3b8': '#475569',
    '#64748b': '#475569',
    '#112240': '#f1f5f9',
    '#1e3a5f': '#e2e8f0'
}

for root, _, files in os.walk(template_dir):
    for file in files:
        if file.endswith('.html'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for old_color, new_color in replacements.items():
                content = content.replace(old_color, new_color)
                # handle case insensitive just in case
                content = content.replace(old_color.upper(), new_color)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
print("Templates updated.")
