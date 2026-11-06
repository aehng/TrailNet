"""Terrain derivatives from a projected, metric digital terrain model."""
import numpy as np
from scipy.ndimage import gaussian_filter


def derive_terrain(dtm, pixel_size_x, pixel_size_y, relief_radius_px=25,
                   azimuth_deg=315.0, altitude_deg=45.0):
    """Return hillshade, slope and local relief as float32 arrays in [0, 1].

    Pixel sizes must be in metres. NaNs are temporarily filled for derivatives
    and restored to zero in outputs. The local relief is elevation minus a
    Gaussian-smoothed surface, scaled robustly by its 2nd/98th percentiles.
    """
    elevation = np.asarray(dtm, dtype=np.float32)
    valid = np.isfinite(elevation)
    if not valid.any():
        raise ValueError("DTM contains no finite elevation cells")
    fill = float(np.nanmedian(elevation))
    z = np.where(valid, elevation, fill)
    dz_dy, dz_dx = np.gradient(z, abs(pixel_size_y), abs(pixel_size_x))
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)
    azimuth = np.deg2rad(azimuth_deg)
    altitude = np.deg2rad(altitude_deg)
    illumination = (np.sin(altitude) * np.cos(slope_rad) +
                    np.cos(altitude) * np.sin(slope_rad) * np.cos(azimuth - aspect))
    hillshade = np.clip((illumination + 1.0) / 2.0, 0.0, 1.0)
    slope = np.clip(slope_rad / (np.pi / 2), 0.0, 1.0)
    local = z - gaussian_filter(z, sigma=max(float(relief_radius_px), 1.0))
    lo, hi = np.percentile(local[valid], [2, 98])
    relief = np.zeros_like(local) if hi <= lo else np.clip((local - lo) / (hi - lo), 0, 1)
    outputs = [hillshade, slope, relief]
    return tuple(np.where(valid, np.nan_to_num(a), 0).astype(np.float32) for a in outputs)
