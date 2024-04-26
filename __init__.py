import openslide
from parser import parse_args
from openslide import OpenSlide
import glob, os
import cv2
import json
import base64
from io import BytesIO
import requests
from PIL import Image
from process_file import ImageProcessor
import numpy as np

class SlideProcessor:
    def __init__(self, slides_path, masks_path, output_path, slide_down_sample_rate = 5):
        self.slides_path = slides_path
        self.masks_path = masks_path
        self.output_path = output_path
        self.slide_down_sample_rate = slide_down_sample_rate
        self.extensions = ['*.svs', '*.tiff', "*.tif", '*.ndpi']

        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def init_image_processor(self, model_dir, tile_size = 1024, overlay_down_sample_rate = 5, gpu_ids=[]):
        self.image_processor = ImageProcessor(model_dir = model_dir, tile_size = 1024, gpu_ids = gpu_ids)
        self.overlay_down_sample_rate = overlay_down_sample_rate

    def open_wsi_slide(self, slide_path):
        try:
            slide = openslide.OpenSlide(slide_path)
            return slide
        except openslide.OpenSlideError as e:
            print(f"Cannot open {slide_path}\nError: {e}")
            return None

    def process_slides(self, save_regions = False):


        slides = [slide for ext in self.extensions for slide in glob.glob(os.path.join(self.slides_path, ext))]
        for slide_path in slides:
            slide = self.open_wsi_slide(slide_path)
            file_name = os.path.basename(slide_path).split('.')[0]
            mask_path = glob.glob(os.path.join(self.masks_path, file_name + "*.png"))

            if len(mask_path) != 1:
                print(f"No/Multiple masks found for {slide_path}")
                continue
            
            mask_path = mask_path[0]
            slide_dimensions = slide.dimensions
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            ratio_width = round(slide_dimensions[0] / mask.shape[1])
            ratio_height = round(slide_dimensions[1] / mask.shape[0])

            if hasattr(self, 'image_processor'):
                heat_map = np.zeros((slide_dimensions[1] // self.overlay_down_sample_rate, slide_dimensions[0] // self.overlay_down_sample_rate, 3), dtype=np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for idx, contour in enumerate(contours):
                x, y, width, height = map(int, cv2.boundingRect(contour))
                # Apply the ratios to the coordinates and dimensions
                x, y, width, height = x * ratio_width, y * ratio_height, width * ratio_width, height * ratio_height
                region = slide.read_region((x, y), 0, (width, height)).resize((width // self.slide_down_sample_rate, height // self.slide_down_sample_rate))

                if hasattr(self, 'image_processor'):
                    results = self.image_processor.test_img(region)

                    x, y, width, height = (x // self.overlay_down_sample_rate,
                    y // self.overlay_down_sample_rate, 
                    width // self.overlay_down_sample_rate, 
                    height // self.overlay_down_sample_rate)
                    heat_map[y:y+height, x:x+width, :] = np.array(results["SegRefined"])

                if (save_regions):
                    if idx == 0:
                        os.makedirs(os.path.join(self.output_path, file_name), exist_ok=True)
                    img_path = os.path.join(self.output_path, file_name, f"{idx+1}_{x}_{y}_{width}_{height}_region.png")
                    region.save(img_path)
            if hasattr(self, 'image_processor'):
                final_heat_map = Image.fromarray(heat_map)
                final_heat_map.save(os.path.join(self.output_path, f"{file_name}_heat_map.png"))


def main():
    args = parse_args()
    processor = SlideProcessor(args.slides_path, 
                               args.masks_path, 
                               args.output_path, 
                               args.slide_down_sample_rate)
    if args.deepliif is not None:
        processor.init_image_processor(args.model_dir, args.tile_size, args.overlay_down_sample_rate)
        processor.process_slides()

if __name__ == "__main__":
    main()