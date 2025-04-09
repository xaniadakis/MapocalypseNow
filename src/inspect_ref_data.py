import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Language option: "en" or "gr" ===
LANG = "gr"

# === Label mappings ===
label_names_en = {
    0: "N/A",
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare land",
    70: "Snow & Ice",
    80: "Water",
    90: "Wetland",
    95: "Mangroves",
    100: "Moss/Lichen"
}
label_names_gr = {
    0: "N/A",
    10: "Δασική κάλυψη",
    20: "Θάμνωση",
    30: "Χορτολίβαδα",
    40: "Καλλιέργιες",
    50: "Δόμηση",
    60: "Γυμνή γη",
    70: "Χιόνι/πάγος",
    80: "Υδάτινες επιφάνειες",
    90: "Υγρότοποι με ποώδη βλάστηση",
    95: "Μαγκρόβια",
    100: "Βρύα/λειχήνες"
}

label_names = label_names_gr if LANG == "gr" else label_names_en

# === Path to reference file ===
cwd = Path(__file__).resolve().parent
data_dir = cwd.parent / "data"
ref_path = data_dir / "GBDA24_ex2_ref_data.tif"

# === Read the data ===
with rasterio.open(ref_path) as src:
    data = src.read(1)

unique, counts = np.unique(data, return_counts=True)
total = counts.sum()

# Filter only known labels
valid = [val in label_names for val in unique]
unique = unique[valid]
counts = counts[valid]

labels = [label_names[val] for val in unique]
percentages = 100 * counts / total

# === Plot ===
plt.figure(figsize=(10, 5))
bars = plt.bar(labels, counts)
plt.title("Συχνότητα Κλάσεων" if LANG == "gr" else "Label Frequency in Reference Raster")
plt.xlabel("Κλάση" if LANG == "gr" else "Class")
plt.ylabel("Πλήθος εικονοστοιχείων" if LANG == "gr" else "Pixel Count")
plt.xticks(rotation=15, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Annotate with percentages
for bar, pct in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + total * 0.01,
             f"{pct:.2f}%",
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
