import os
import pandas as pd

records_dir = "data/records"

rows = []

for fname in os.listdir(records_dir):
    if fname.endswith(".hea"):
        path = os.path.join(records_dir, fname)
        record = fname.replace(".hea","")

        with open(path) as f:
            lines = f.readlines()

        dx = None
        for line in lines:
            if line.startswith("#Dx"):
                dx = line.split(":")[1].strip()
                break

        if dx is None:
            continue

        labels = dx.replace(",", "|")

        rows.append({
            "record": f"data/records/{record}",
            "labels": labels
        })

df = pd.DataFrame(rows)
df.to_csv("data/records.csv", index=False)
print("Saved data/records.csv with", len(df), "records")
