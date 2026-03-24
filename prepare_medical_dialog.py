#!/usr/bin/env python3
"""
Download knowrohit07/know_medical_dialogue_v2 from HuggingFace and write
it to medical_dialog.txt so it can be used with train.py.

Note: UCSD26/medical_dialog was replaced because it uses an old dataset
script format (medical_dialog.py) that is no longer supported in
datasets >= 3.0.
"""

from datasets import load_dataset

print("Loading knowrohit07/know_medical_dialogue_v2 ...")
ds = load_dataset("knowrohit07/know_medical_dialogue_v2")

# Print columns so we know what we're working with
split0 = list(ds.keys())[0]
print(f"Columns: {ds[split0].column_names}")
print(f"First row sample: {ds[split0][0]}")

lines = []
for split in ds.values():
    for row in split:
        # Dataset has 'instruction' (patient) and 'output' (doctor) fields
        context = row.get("instruction", "").strip()
        response = row.get("output", "").strip()
        if context:
            lines.append(context)
        if response:
            lines.append(response)
        lines.append("")   # blank line between dialogs

content = "\n".join(lines)

out_path = "medical_dialog.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Wrote {len(content):,} characters to {out_path}")
print(f"Unique chars (vocab size): {len(set(content))}")
