import os

app_path = r"d:\ML\RideRush\RideRush\app.py"

with open(app_path, "r", encoding="utf-8") as f:
    app_code = f.read()

app_code = app_code.replace("alpha=.25", "alpha=0.7")
app_code = app_code.replace("alpha=.3", "alpha=0.7")

with open(app_path, "w", encoding="utf-8") as f:
    f.write(app_code)

print("Alpha updated.")
