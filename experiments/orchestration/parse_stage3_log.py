import json

with open("research/kaggle_stage3_execution.log", "r", encoding="utf-8") as f:
    content = f.read()

# The log file is formatted as json lines or json array chunks
lines = content.splitlines()
all_text = []
for line in lines:
    line = line.strip().lstrip(",")
    if not line:
        continue
    try:
        obj = json.loads(line)
        if isinstance(obj, list):
            for item in obj:
                all_text.append(item.get("data", ""))
        elif isinstance(obj, dict):
            all_text.append(obj.get("data", ""))
    except Exception:
        pass

combined = "".join(all_text)
with open("research/kaggle_stage3_readable.txt", "w", encoding="utf-8") as out:
    out.write(combined)

print("Saved readable log. Total length:", len(combined))

# Find post-training audit outputs
idx = combined.find("[AUDIT")
if idx != -1:
    print("\n--- POST TRAINING AUDIT FOUND ---")
    print(combined[idx:idx+3000])
else:
    print("Post-training audit marker not found in standard search.")
