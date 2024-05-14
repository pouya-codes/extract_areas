
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
        self.opt.gpu_ids = gpu_ids  # overwrite gpu_ids; for test command, default gpu_ids at first is [] which will be translated to a list of all gpus
        # print_options(self.opt)
        

    # def test(self, img_path, eager_mode=True, color_dapi=True, color_marker=True):
    #     img = Image.open(img_path).convert('RGB')
    #     images, scoring = infer_modalities(img, self.tile_size, self.model_dir, eager_mode, color_dapi, color_marker, self.opt)
    #     filename = os.path.basename(img_path)
    #     results = {}
    #     for name, i in images.items():
    #         results[name] = i
    #     return results
            # i.save(os.path.join(
            #     self.output_dir,
            #     filename.replace('.' + filename.split('.')[-1], f'_{name}.png')
            # ))

        # if scoring is not None:
        #     with open(os.path.join(
        #             self.output_dir,
        #             filename.replace('.' + filename.split('.')[-1], f'.json')
        #     ), 'w') as f:
        #         json.dump(scoring, f, indent=2)
        
    def test_img(self, img, eager_mode=False, color_dapi=False, color_marker=False):
        img = img.convert('RGB')
        images, scoring = infer_modalities(img, self.tile_size, self.model_dir, eager_mode, color_dapi, color_marker, self.opt)
        if (self.post_processing):
            # https://github.com/nadeemlab/DeepLIIF?tab=readme-ov-file#cloud-api-endpoints
            images, scoring = postprocess(img, images, self.tile_size, 'DeepLIIF', seg_thresh=150, size_thresh='auto', marker_thresh='auto', size_thresh_upper=None)
        results = {}
        for name, i in images.items():
            results[name] = i
        return results

