"""Read whole-slide image regions straight from disk.

The API used to receive regions as JPEG uploads because the caller lived on a
different host. Now that the viewer and this service share a machine and a
filesystem, the caller sends a slide path plus level-0 coordinates instead, and
we pull the pixels from the nearest pyramid level. A 4x request against a slide
with a 4x level reads a quarter of the pixels in each axis rather than reading
level 0 and throwing 15/16ths of it away.
"""

import math
import os
from pathlib import Path

from PIL import Image, ImageDraw

# Pillow refuses very large images by default as a decompression-bomb guard.
# Slide regions are legitimately large and come from a trusted local path.
Image.MAX_IMAGE_PIXELS = None


class SlideAccessError(Exception):
    """Raised when a slide path is unusable or outside the permitted roots."""


def resolve_slide_path(raw_path, allowed_roots):
    """Resolve ``raw_path`` and confirm it sits inside one of ``allowed_roots``.

    The endpoint takes a filesystem path from an HTTP body, so without this the
    service would happily open any file the process can read. Symlinks are
    resolved first: OMERO's ManagedRepository stores in-place imports as
    symlinks into a staging area, so the link and its target must both be
    checked against the allowlist rather than trusting the link's own location.
    """
    if not raw_path or not isinstance(raw_path, str):
        raise SlideAccessError("slide_path is required")

    if not allowed_roots:
        raise SlideAccessError("no slide roots are configured on the server")

    candidate = Path(os.path.realpath(raw_path))
    roots = [Path(os.path.realpath(root)) for root in allowed_roots]

    if not any(candidate == root or root in candidate.parents for root in roots):
        raise SlideAccessError(f"slide path is outside the permitted roots: {raw_path}")

    if not candidate.is_file():
        raise SlideAccessError(f"slide file does not exist: {raw_path}")
    if not os.access(candidate, os.R_OK):
        raise SlideAccessError(f"slide file is not readable by the service: {raw_path}")

    return candidate


def clamp_bbox(bbox, slide_size):
    """Clip a level-0 ``[x, y, w, h]`` box to the slide, or raise if it misses."""
    try:
        x0, y0, w, h = (int(round(float(v))) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise SlideAccessError(f"roi_bbox must be four numbers: {bbox}") from exc

    slide_w, slide_h = slide_size
    x0 = max(0, min(x0, slide_w - 1))
    y0 = max(0, min(y0, slide_h - 1))
    w = min(w, slide_w - x0)
    h = min(h, slide_h - y0)

    if w <= 0 or h <= 0:
        raise SlideAccessError(f"roi_bbox does not overlap the slide: {bbox}")

    return x0, y0, w, h


def level0_mpp(slide):
    """Microns per pixel at level 0, or ``None`` when the slide does not say.

    Falls back to the objective power because some scanners record only that.
    Models that pick a tile size from physical scale need this; without it they
    have to assume a magnification and will mis-size their detections.
    """
    for key in ("openslide.mpp-x", "openslide.mpp-y"):
        try:
            value = float(slide.properties.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value

    try:
        objective = float(slide.properties.get("openslide.objective-power"))
    except (TypeError, ValueError):
        return None
    # 40x scanners resolve about 0.25 um/px, and the rest scale linearly.
    return 10.0 / objective if objective > 0 else None


def read_region(slide_path, bbox, downsample=1.0, max_pixels=None):
    """Read a level-0 ``bbox`` at ``downsample``, reading from the best level.

    Returns ``(image, info)`` where ``image`` is RGB and ``info`` records the
    downsample actually applied. The applied value can exceed the request when
    ``max_pixels`` forces a coarser read, so callers must scale coordinates by
    ``info["downsample"]`` rather than by what they asked for.
    """
    # Imported here so the path/polygon helpers above stay usable (and
    # testable) on hosts without the OpenSlide shared library.
    import openslide

    downsample = max(1.0, float(downsample or 1.0))
    slide = openslide.OpenSlide(str(slide_path))
    try:
        x0, y0, w, h = clamp_bbox(bbox, slide.dimensions)
        base_mpp = level0_mpp(slide)

        # A region the caller asked for at full resolution can be far larger
        # than anything a model should see at once. Coarsen rather than trying
        # to allocate it and dying with an OOM mid-request.
        if max_pixels:
            requested = (w / downsample) * (h / downsample)
            if requested > max_pixels:
                downsample *= math.sqrt(requested / max_pixels)

        target_w = max(1, int(round(w / downsample)))
        target_h = max(1, int(round(h / downsample)))

        level = slide.get_best_level_for_downsample(downsample)
        level_downsample = slide.level_downsamples[level]
        read_w = max(1, int(round(w / level_downsample)))
        read_h = max(1, int(round(h / level_downsample)))

        # read_region takes its location in level-0 coordinates regardless of
        # the level being read, so x0/y0 are passed through unscaled.
        tile = slide.read_region((x0, y0), level, (read_w, read_h))
    finally:
        slide.close()

    if tile.mode == "RGBA":
        # Slide edges and scanner padding come back transparent; flatten onto
        # white so models see the same background the viewer shows.
        flattened = Image.new("RGB", tile.size, (255, 255, 255))
        flattened.paste(tile, mask=tile.split()[3])
        tile.close()
        tile = flattened
    else:
        tile = tile.convert("RGB")

    if tile.size != (target_w, target_h):
        resized = tile.resize((target_w, target_h), Image.Resampling.LANCZOS)
        tile.close()
        tile = resized

    applied_downsample = w / target_w if target_w else downsample
    info = {
        "bbox": [x0, y0, w, h],
        "downsample": applied_downsample,
        "level": level,
        "level_downsample": level_downsample,
        "size": [target_w, target_h],
        "mpp": base_mpp * applied_downsample if base_mpp else None,
    }
    return tile, info


def polygon_mask(polygon, bbox, size):
    """Rasterise a level-0 polygon into a region-local binary mask.

    ``size`` is the region's pixel size after downsampling; the scale is derived
    from it rather than taken as an argument so the mask cannot drift from the
    image when a read was coarsened by ``max_pixels``.
    """
    width, height = size
    x0, y0, w, h = bbox
    scale_x = width / w if w else 1.0
    scale_y = height / h if h else 1.0

    points = []
    for point in polygon or []:
        if isinstance(point, dict):
            px, py = point.get("x", point.get("X")), point.get("y", point.get("Y"))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            px, py = point[0], point[1]
        else:
            continue
        if px is None or py is None:
            continue
        try:
            lx = (float(px) - x0) * scale_x
            ly = (float(py) - y0) * scale_y
        except (TypeError, ValueError):
            continue
        points.append(
            (
                max(0, min(width - 1, int(round(lx)))),
                max(0, min(height - 1, int(round(ly)))),
            )
        )

    mask = Image.new("L", (width, height), 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, outline=255, fill=255)
    else:
        # No usable polygon: process the whole box rather than nothing at all.
        mask.paste(255, (0, 0, width, height))
    return mask
