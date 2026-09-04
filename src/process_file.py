import os
import time

import numpy as np
import torch
from PIL import Image
from deepliif.models import inference, init_nets, postprocess
from deepliif.options import Options
from deepliif.postprocessing import compute_final_results

# DeepLIIF was trained on 512 px tiles of 40x tissue, so one tile covers about
# 128 um of slide. Tiles handed to it are resized to opt.scale_size (512), which
# means the tile size we cut at is really a statement about magnification: cut
# 128 um worth of pixels and the network sees what it was trained on. Getting
# this wrong is not just slow, it changes the cell counts, because the
# postprocessor picks its size thresholds from the tile size.
TILE_MICRONS = 128.0

# Everything downstream reads SegRefined (and Marker, for scoring). The other
# three modalities are generated, converted to PIL and thrown away, so ask
# inference to skip them.
SEG_ONLY = True


def _resolution_for_mpp(mpp, tile_size):
    """Magnification bucket to threshold cell sizes against.

    The postprocessor measures cells in pixels of the image it is handed, so
    the bucket has to follow the region's real resolution. DeepLIIF infers it
    from the tile size instead, which only holds while tiles are always cut at
    full resolution; once a region is read downsampled the two disagree.
    """
    if mpp and mpp > 0:
        if mpp <= 0.35:
            return '40x'
        return '20x' if mpp <= 0.71 else '10x'
    return '40x' if tile_size > 384 else ('20x' if tile_size > 192 else '10x')


def _as_array(mask):
    """Accept a PIL mask or an array and hand back a single-channel array."""
    if mask is None or isinstance(mask, np.ndarray):
        return mask
    if isinstance(mask, Image.Image):
        return np.asarray(mask.convert('L'))
    return np.asarray(mask)


class _HalfPrecisionNet(torch.nn.Module):
    """Run a generator in fp16 while still accepting and returning fp32.

    The nets are TorchScript modules called from Dask worker threads, so
    torch.autocast is no good here: its state is thread-local and never reaches
    the forward pass. Converting the weights and casting at the boundary works
    regardless of which thread runs it, and on a T4 it is worth roughly 2.7x.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net.half()

    def forward(self, x):
        return self.net(x.half()).float()


class ImageProcessor:
    def __init__(self, model_dir, tile_size=512, post_processing=True, gpu_ids=[], use_fp16=True):

        self.model_dir = model_dir
        self.tile_size = tile_size
        self.post_processing = post_processing

        files = os.listdir(self.model_dir)
        assert 'train_opt.txt' in files, f'file train_opt.txt is missing from model directory {self.model_dir}'

        self.opt = Options(path_file=os.path.join(self.model_dir, 'train_opt.txt'), mode='test')
        # Enable DataParallel so models can utilize all provided GPUs during inference
        self.opt.use_dp = True
        number_of_gpus_all = torch.cuda.device_count()
        if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
            # number_of_gpus = 0
            gpu_ids = [-1]
            print(
                f'Specified to use GPU {self.opt.gpu_ids} for inference, but there are only {number_of_gpus_all} GPU devices. Switched to CPU inference.')

        if len(gpu_ids) > 0 and gpu_ids[0] == -1:
            gpu_ids = []
        elif len(gpu_ids) == 0:
            gpu_ids = list(range(number_of_gpus_all))
        print(f'Using GPU {gpu_ids} for inference.')
        self.opt.gpu_ids = gpu_ids

        self.use_fp16 = bool(use_fp16) and len(gpu_ids) > 0
        self._load_nets()

    def _load_nets(self):
        """Load the generators up front, in fp16 when running on a GPU.

        init_nets is lru_cached and inference() reaches for the same entry, so
        replacing the values in the dict it returns is enough to make every
        later call use the converted nets. Doing it here also means the first
        request does not pay for loading eleven TorchScript modules.
        """
        model_dir = os.getenv('DEEPLIIF_MODEL_DIR', self.model_dir)
        nets = init_nets(model_dir, False, self.opt)
        if not self.use_fp16:
            return
        for name, net in list(nets.items()):
            if not isinstance(net, _HalfPrecisionNet):
                nets[name] = _HalfPrecisionNet(net)

    def tile_size_for_mpp(self, mpp, downsample=1.0):
        """Tile size, in region pixels, for a region read at `downsample`.

        A tile covers TILE_MICRONS of tissue at full resolution. Asking for a
        downsampled region is asking to trade detail for speed, so the tile
        keeps its pixel size and covers `downsample` times more tissue
        instead: the tile count then falls with the region's pixel count,
        which is the entire point of downsampling. Sizing tiles off the
        region's own mpp would keep every tile at TILE_MICRONS and so keep the
        tile count -- and the runtime -- constant no matter how far the region
        was downsampled.

        Returns the configured default when the slide has no resolution
        metadata, which is the best we can do without guessing.
        """
        if not mpp or mpp <= 0:
            return self.tile_size
        target = TILE_MICRONS * max(float(downsample), 1.0) / float(mpp)
        # Snap to a multiple of 32 so the overlap (tile_size // 16) stays even.
        snapped = int(round(target / 32.0)) * 32
        return max(64, min(1024, snapped))

    def test_img(self, img, eager_mode=False, color_dapi=False, color_marker=False,
                 patch_classifier_mask=None, tissue_mask=None, tile_size=None,
                 mpp=None):
        img = img.convert('RGB')
        tile_size = int(tile_size or self.tile_size)
        # The postprocessor indexes and thresholds these as arrays.
        tissue_mask = _as_array(tissue_mask)
        patch_classifier_mask = _as_array(patch_classifier_mask)

        # The colour transforms only touch DAPI/Marker modality images, which
        # seg-only inference does not produce.
        seg_only = SEG_ONLY and not (color_dapi or color_marker)

        started = time.perf_counter()
        images = inference(
            img,
            tile_size=tile_size,
            overlap_size=tile_size // 16,
            model_path=self.model_dir,
            eager_mode=eager_mode,
            color_dapi=color_dapi,
            color_marker=color_marker,
            opt=self.opt,
            seg_only=seg_only,
        )
        inference_seconds = time.perf_counter() - started

        resolution = _resolution_for_mpp(mpp, tile_size)

        scoring = {}
        if not hasattr(self.opt, 'seg_gen') or self.opt.seg_gen:
            started = time.perf_counter()
            if self.opt.model == 'DeepLIIF':
                # Same work as postprocess(), except that it derives the
                # resolution from the tile size and we already know it.
                overlay, refined, scoring = compute_final_results(
                    img, images['Seg'], images.get('Marker'), resolution,
                    patch_classifier_mask=patch_classifier_mask,
                    tissue_mask=tissue_mask,
                )
                images = {
                    **images,
                    'SegOverlaid': Image.fromarray(overlay),
                    'SegRefined': Image.fromarray(refined),
                }
            else:
                post_images, scoring = postprocess(
                    img, images, tile_size, self.opt.model,
                    patch_classifier_mask=patch_classifier_mask,
                    tissue_mask=tissue_mask,
                )
                images = {**images, **post_images}
            scoring = dict(scoring or {})
            scoring['postprocess_seconds'] = round(time.perf_counter() - started, 3)

        scoring['inference_seconds'] = round(inference_seconds, 3)
        scoring['tile_size'] = tile_size
        scoring['resolution'] = resolution
        scoring['precision'] = 'fp16' if self.use_fp16 else 'fp32'

        return dict(images), scoring
