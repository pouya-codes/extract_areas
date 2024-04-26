
import os
import os
from deepliif.options import Options, print_options
from deepliif.util.visualizer import save_images
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from deepliif.models import infer_modalities
import json
class ImageProcessor:
    def __init__(self, model_dir, tile_size=512, gpu_ids=[]):

        self.model_dir = model_dir
        self.tile_size = tile_size

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

        self.opt.gpu_ids = gpu_ids  # overwrite gpu_ids; for test command, default gpu_ids at first is [] which will be translated to a list of all gpus
        # print_options(self.opt)
        

    def test(self, img_path, eager_mode=True, color_dapi=True, color_marker=True):
        img = Image.open(img_path).convert('RGB')
        images, scoring = infer_modalities(img, self.tile_size, self.model_dir, eager_mode, color_dapi, color_marker, self.opt)
        filename = os.path.basename(img_path)
        results = {}
        for name, i in images.items():
            results[name] = i
        return results
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
        
    def test_img(self, img, eager_mode=True, color_dapi=True, color_marker=True):
        img = img.convert('RGB')
        images, scoring = infer_modalities(img, self.tile_size, self.model_dir, eager_mode, color_dapi, color_marker, self.opt)
        results = {}
        for name, i in images.items():
            results[name] = i
        return results
    
# if __name__ == "__main__":
#     # Initialize image processor
#     image_processor = ImageProcessor(model_dir="/home/pouya/Develop/UBC/QA-QC/Codes/Models/DeepLIIF_Latest_Model", 
#                                      output_dir="/home/pouya/Develop/UBC/QA-QC/Datasets/temp_out/",
#                                      tile_size=512)
#     # Load image
#     image_path = "/home/pouya/Develop/UBC/QA-QC/Datasets/temp/braf 207_1_region.png"
#     image_processor.test(image_path)
