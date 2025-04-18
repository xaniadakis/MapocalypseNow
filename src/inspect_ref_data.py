import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Language option: "en" or "gr" ===
LANG = "en"

# === Label mappings ===
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
#
# # === Plot ===
# plt.figure(figsize=(10, 5))
# bars = plt.bar(labels, counts)
# plt.title("Συχνότητα Κλάσεων" if LANG == "gr" else "Label Frequency in Reference Raster")
# plt.xlabel("Κλάση" if LANG == "gr" else "Class")
# plt.ylabel("Πλήθος εικονοστοιχείων" if LANG == "gr" else "Pixel Count")
# plt.xticks(rotation=15, ha="right")
# plt.grid(axis="y", linestyle="--", alpha=0.5)
#
# # Annotate with percentages
# # for bar, pct in zip(bars, percentages):
# #     plt.text(bar.get_x() + bar.get_width() / 2,
# #              bar.get_height() + total * 0.01,
# #              f"{pct:.2f}%",
# #              ha='center', va='bottom', fontsize=9)
#
# plt.tight_layout()
# plt.show()

import seaborn as sns
sns.set_theme(style="darkgrid", palette="muted", font_scale=0.9)

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# === Plot ===
# plt.style.use("seaborn-whitegrid")  # Clean and professional
fig, ax = plt.subplots(figsize=(12, 6))

# Bar colors: optional custom color palette
# colors = plt.cm.tab20c(np.linspace(0.1, 0.9, len(labels)))
# bars = ax.bar(labels, counts, color=colors, edgecolor="black")
palette = sns.color_palette("deep", n_colors=len(labels))
bars = ax.bar(labels, counts, color=palette, edgecolor="black")

# Titles and labels
title = "Συχνότητα Κλάσεων" if LANG == "gr" else "Label Frequency in Reference Raster"
xlabel = "Κλάση" if LANG == "gr" else "Class"
ylabel = "Πλήθος εικονοστοιχείων" if LANG == "gr" else "Pixel Count"

# ax.set_title(title, fontsize=15, weight="bold", pad=20)
# ax.set_xlabel(xlabel, fontsize=14)
ax.set_ylabel(ylabel, fontsize=14)

# X-axis
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)

# Y-axis
from matplotlib.ticker import EngFormatter
ax.yaxis.set_major_formatter(EngFormatter(unit=""))

# Grid styling
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

# Annotate bars with percentages
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.annotate(f"{pct:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, color="black")

# Final layout
plt.tight_layout()
plt.show()

