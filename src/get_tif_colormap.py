from pathlib import Path
import rasterio
from v_prepare_training_data import label_to_cls, label_to_text  # adjust import path if needed

def rgb_ansi(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

cwd = Path(__file__).resolve().parent
data_dir = cwd.parent / "data"
processed_dir = data_dir / "processed"
tif_path = processed_dir / "GBDA24_ex2_34SEH_ref_data_reprojected.tif"

cls_to_label = {v: k for k, v in label_to_cls.items()}

with rasterio.open(tif_path) as src:
    cmap = src.colormap(1)

print(f"{'Model Class':<12} | {'Label':<5} | {'Class Name':<20} | {'RGB':<13} | {'HEX'}")
print("-" * 70)

for cls_idx in sorted(cls_to_label.keys()):
    label = cls_to_label[cls_idx]
    name = label_to_text[label]

    if label in cmap:
        r, g, b = cmap[label][:3]
        hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
        rgb_str = f"{r},{g},{b}"
        colored_name = rgb_ansi(r, g, b, f"{name:<20}")
    else:
        r, g, b = 0, 0, 0
        hex_color = "N/A"
        rgb_str = "N/A"
        colored_name = f"{name:<20}"

    print(f"{cls_idx:<12} | {label:<5} | {colored_name} | {rgb_str:<13} | {hex_color}")
