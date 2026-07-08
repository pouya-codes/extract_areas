import os
import cv2
import numpy as np
from tqdm import tqdm
import openslide

# =========================
# HistomicsTK
# =========================
from histomicstk.saliency.tissue_detection import get_tissue_mask

# =========================
# TIAToolbox
# =========================
from tiatoolbox.tools.tissuemask import OtsuTissueMasker, MorphologicalMasker
from tiatoolbox.wsicore.wsireader import WSIReader

# =========================
# Slideflow
# =========================
import slideflow as sf
from slideflow.slide import qc


# -------------------------------------------------
# Utils
# -------------------------------------------------
def save_mask(mask, path):
    mask = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(path, mask)


def get_thumbnail_openslide(wsi_path, size=2048):
    slide = openslide.OpenSlide(wsi_path)
    thumb = slide.get_thumbnail((size, size))
    return np.array(thumb)


# -------------------------------------------------
# HistomicsTK mask
# -------------------------------------------------
def mask_histomicstk(img):
    mask, _ = get_tissue_mask(img, deconvolve_first=True)
    return mask


# -------------------------------------------------
# TIAToolbox masks
# -------------------------------------------------
def mask_tiatoolbox(img):
    otsu_masker = OtsuTissueMasker()
    morph_masker = MorphologicalMasker()

    mask_otsu = otsu_masker.fit_transform([img])[0]
    mask_morph = morph_masker.fit_transform([img])[0]

    return mask_otsu, mask_morph


# -------------------------------------------------
# Slideflow mask
# -------------------------------------------------
def mask_slideflow(wsi_path, tile_px=256, tile_um=256):
    # Slideflow WSI now requires tile geometry; use sane defaults for masking.
    slide = sf.WSI(wsi_path, tile_px=tile_px, tile_um=tile_um)
    mask = slide.qc(qc.Otsu())
    return mask


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def process_folder(wsi_dir, out_dir, tile_px=256, tile_um=256):

    os.makedirs(out_dir, exist_ok=True)

    wsi_files = [
        f for f in os.listdir(wsi_dir)
        if f.lower().endswith((".svs", ".tif", ".ndpi", ".mrxs"))
    ]

    for fname in tqdm(wsi_files):
        path = os.path.join(wsi_dir, fname)
        base = os.path.splitext(fname)[0]

        slide_out = os.path.join(out_dir, base)
        os.makedirs(slide_out, exist_ok=True)

        # -----------------------------------------
        # Thumbnail for HistomicsTK + TIAToolbox
        # -----------------------------------------
        img = get_thumbnail_openslide(path, size=2048)

        # -----------------------------------------
        # HistomicsTK
        # -----------------------------------------
        htk_mask = mask_histomicstk(img)
        save_mask(htk_mask, f"{slide_out}/mask_histomicstk.png")

        # -----------------------------------------
        # TIAToolbox
        # -----------------------------------------
        mask_otsu, mask_morph = mask_tiatoolbox(img)
        save_mask(mask_otsu, f"{slide_out}/mask_tia_otsu.png")
        save_mask(mask_morph, f"{slide_out}/mask_tia_morph.png")

        # -----------------------------------------
        # Slideflow
        # -----------------------------------------
        try:
            sf_mask = mask_slideflow(path, tile_px=tile_px, tile_um=tile_um)
            save_mask(sf_mask, f"{slide_out}/mask_slideflow.png")
        except Exception as e:
            print(f"Slideflow failed for {fname}: {e}")


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":

    WSI_FOLDER = "/home/aimlab/Downloads/PDA"
    OUTPUT_FOLDER = "/home/aimlab/Downloads/PDA/masks_comparison"

    process_folder(WSI_FOLDER, OUTPUT_FOLDER)
