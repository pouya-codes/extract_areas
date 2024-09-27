
import os
from deepliif.options import Options, print_options
from deepliif.util.visualizer import save_images
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from deepliif.models import infer_modalities, postprocess

class ImageProcessor:
    def __init__(self, model_dir, tile_size=512, post_processing=True, gpu_ids=[]):

        self.model_dir = model_dir
        self.tile_size = tile_size
        self.post_processing = post_processing

        files = os.listdir(self.model_dir)
        assert 'train_opt.txt' in files, f'file train_opt.txt is missing from model directory {self.model_dir}'
        
        self.opt = Options(path_file=os.path.join(self.model_dir, 'train_opt.txt'), mode='test')
        self.opt.use_dp = False
        number_of_gpus_all = torch.cuda.device_count()
        if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
            number_of_gpus = 0
            gpu_ids = [-1]
            print(
                f'Specified to use GPU {self.opt.gpu_ids} for inference, but there are only {number_of_gpus_all} GPU devices. Switched to CPU inference.')

        if len(gpu_ids) > 0 and gpu_ids[0] == -1:
            gpu_ids = []
        elif len(gpu_ids) == 0:
            gpu_ids = list(range(number_of_gpus_all))
        print(f'Using GPU {gpu_ids} for inference.')
        self.opt.gpu_ids = gpu_ids  
        

        
    def test_img(self, img, eager_mode=False, color_dapi=False, color_marker=False, patch_classifier_mask = None):
        img = img.convert('RGB')
        if patch_classifier_mask is not None:
            images, scoring = infer_modalities(img, self.tile_size, self.model_dir, eager_mode, color_dapi, color_marker, self.opt, patch_classifier_mask)
        else:
            images, scoring = infer_modalities(img, self.tile_size, self.model_dir, eager_mode, color_dapi, color_marker, self.opt)
        # if (self.post_processing):
            # https://github.com/nadeemlab/DeepLIIF?tab=readme-ov-file#cloud-api-endpoints
            # images, scoring = postprocess(img, images, self.tile_size, 'DeepLIIF', seg_thresh=150, size_thresh='auto', marker_thresh='auto', size_thresh_upper=None)
        results = {}
        for name, i in images.items():
            results[name] = i
        return results, scoring

