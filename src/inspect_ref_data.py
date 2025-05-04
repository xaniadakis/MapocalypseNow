import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LANG = "en"

label_names_en = {
    0: "N/A",
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / Sparse vegetation",
    70: "Snow and Ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and Lichen"
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

cwd = Path(__file__).resolve().parent
data_dir = cwd.parent / "data"
ref_path = data_dir / "GBDA24_ex2_ref_data.tif"

with rasterio.open(ref_path) as src:
    data = src.read(1)

unique, counts = np.unique(data, return_counts=True)
total = counts.sum()

valid = [val in label_names for val in unique]
unique = unique[valid]
counts = counts[valid]

labels = [label_names[val] for val in unique]
percentages = 100 * counts / total

import seaborn as sns
sns.set_theme(style="darkgrid", palette="muted", font_scale=0.9)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

palette = sns.color_palette("deep", n_colors=len(labels))
bars = ax.bar(labels, counts, color=palette, edgecolor="black")

title = "Συχνότητα Κλάσεων" if LANG == "gr" else "Label Frequency in Reference Raster"
xlabel = "Κλάση" if LANG == "gr" else "Class"
ylabel = "Πλήθος εικονοστοιχείων" if LANG == "gr" else "Pixel Count"

ax.set_ylabel(ylabel, fontsize=14)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)

from matplotlib.ticker import EngFormatter
ax.yaxis.set_major_formatter(EngFormatter(unit=""))

ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.annotate(f"{pct:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, color="black")

plt.tight_layout()
plt.show()

