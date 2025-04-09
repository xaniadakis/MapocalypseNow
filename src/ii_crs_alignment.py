from pathlib import Path
import rasterio
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from rasterio.warp import (
    transform_bounds,
    reproject,
    Resampling,
)
from rasterio.transform import from_bounds

cwd = Path(__file__).resolve().parent
data_dir = cwd.parent / "data"
processed_dir = data_dir / "processed"

all_boxes = []
crs = None

for i, tif in enumerate(processed_dir.glob("*/*.tif")):
    with rasterio.open(tif) as src:
        if i == 0:
            crs = src.crs
        bounds = src.bounds
        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        all_boxes.append(geom)

merged_geom = unary_union(all_boxes)
merged_bounds = merged_geom.bounds
geojson_geom = mapping(merged_geom)

val_tif_path = data_dir / "GBDA24_ex2_ref_data.tif"
val_reprojected_path = processed_dir / "GBDA24_ex2_ref_data_reprojected.tif"

with rasterio.open(val_tif_path) as val_src:
    val_crs = val_src.crs
    val_bounds = val_src.bounds

    val_bounds_utm = transform_bounds(
        val_crs,
        crs,
        val_bounds.left,
        val_bounds.bottom,
        val_bounds.right,
        val_bounds.top,
    )

print("========= TRAINING DATA =========")
print(f"CRS:      {crs.to_string()}")
print(
    f"Bounds:   X [{merged_bounds[0]:.0f}, {merged_bounds[2]:.0f}], "
    f"Y [{merged_bounds[1]:.0f}, {merged_bounds[3]:.0f}]"
)
print(
    f"Size:     {(merged_bounds[2] - merged_bounds[0]) / 1000:.3f} km wide × "
    f"{(merged_bounds[3] - merged_bounds[1]) / 1000:.3f} km tall"
)

print(f"\n======== VALIDATION DATA ========")
print(f"Original CRS:  {val_crs.to_string()}")
print(f"Target CRS:    {crs.to_string()}")
print(
    f"Bounds:        X [{val_bounds_utm[0]:.0f}, {val_bounds_utm[2]:.0f}], "
    f"Y [{val_bounds_utm[1]:.0f}, {val_bounds_utm[3]:.0f}]"
)
print(
    f"Size:          {(val_bounds_utm[2] - val_bounds_utm[0]) / 1000:.3f} km wide × "
    f"{(val_bounds_utm[3] - val_bounds_utm[1]) / 1000:.3f} km tall"
)

# reproject label .tif with 10m resolution
target_resolution = 10  # meters
width = int((merged_bounds[2] - merged_bounds[0]) / target_resolution)
height = int((merged_bounds[3] - merged_bounds[1]) / target_resolution)
target_transform = from_bounds(*merged_bounds, width=width, height=height)

with rasterio.open(val_tif_path) as src:
    new_meta = src.meta.copy()
    new_meta.update(
        {
            "crs": crs,
            "transform": target_transform,
            "width": width,
            "height": height,
            'count': src.count,
            'dtype': src.dtypes[0],  # in case you're working with specific dtype
            'nodata': 0
        }
    )

    if src.count == 1 and src.colormap(1):
        new_meta['photometric'] = 'palette'
        colormap = src.colormap(1)
    else:
        colormap = None

    with rasterio.open(val_reprojected_path, "w", **new_meta) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=crs,
                resampling=Resampling.nearest,
            )
        if colormap:
            dst.write_colormap(1, colormap)

with rasterio.open(val_reprojected_path) as reproj:
    reproj_bounds = reproj.bounds

    print(f"\n======== REPROJECTED VALIDATION TIFF INFO ========")
    print(f"CRS:         {reproj.crs}")
    print(f"Bounds:      {reproj.bounds}")
    print(f"Size:        {reproj.width} x {reproj.height}")
    print(f"Resolution:  {reproj.res}")
    print(f"Transform:   {reproj.transform}")
    print(
        f"Size:          {(reproj_bounds[2] - reproj_bounds[0]) / 1000:.3f} km wide × "
        f"{(reproj_bounds[3] - reproj_bounds[1]) / 1000:.3f} km tall"
    )
