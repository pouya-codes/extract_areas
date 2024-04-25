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
    def __init__(self, slides_path, masks_path, output_path, down_sample_rate_slide, down_sample_rate_mask=5):
        self.slides_path = slides_path
        self.masks_path = masks_path
        self.output_path = output_path
        self.down_sample_rate_slide = down_sample_rate_slide
        self.down_sample_rate_mask = down_sample_rate_mask
        self.extensions = ['*.svs', '*.tiff', "*.tif", '*.ndpi']
        self.image_processor = ImageProcessor(model_dir="/home/pouya/Develop/UBC/QA-QC/Codes/Models/DeepLIIF_Latest_Model", 
                                     output_dir="/home/pouya/Develop/UBC/QA-QC/Datasets/temp_out/",
                                     tile_size=1024)
        # Load image
        # image_path = "/home/pouya/Develop/UBC/QA-QC/Datasets/temp/braf 207_1_region.png"
        # self.image_processor.test(image_path)

    def open_wsi_slide(self, slide_path):
        try:
            slide = openslide.OpenSlide(slide_path)
            return slide
        except openslide.OpenSlideError as e:
            print(f"Error: {e}")
            return None

    def process_slides(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

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

            heat_map = np.zeros((slide_dimensions[1] // self.down_sample_rate_mask, slide_dimensions[0] // self.down_sample_rate_mask, 3), dtype=np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for idx, contour in enumerate(contours):
                if idx == 0:
                    os.makedirs(os.path.join(self.output_path, file_name), exist_ok=True)

                x, y, width, height = map(int, cv2.boundingRect(contour))
                # Apply the ratios to the coordinates and dimensions
                x, y, width, height = x * ratio_width, y * ratio_height, width * ratio_width, height * ratio_height
                region = slide.read_region((x, y), 0, (width, height)).resize((width // self.down_sample_rate_slide, height // self.down_sample_rate_slide))
                
                results = self.image_processor.test_img(region)

                x, y, width, height = (x // self.down_sample_rate_mask,
                y // self.down_sample_rate_mask, 
                width // self.down_sample_rate_mask, 
                height // self.down_sample_rate_mask)

                heat_map[y:y+height, x:x+width, :] = np.array(results["SegRefined"])
                # img_path = os.path.join(self.output_path, file_name, f"{idx+1}_{x}_{y}_{width}_{height}_region.png")
                # print(f"Saving region to {img_path}")
                # region.save(img_path)
                final_heat_map = Image.fromarray(heat_map)
                final_heat_map.save(os.path.join(self.output_path, f"{file_name}_heat_map.png"))
                # exit()


def main():
    args = parse_args()
    SLIDE_PATH="/home/pouya/Develop/UBC/QA-QC/Datasets/R204brafv600e/Slides"
    MASKS_PATH="/home/pouya/Develop/UBC/QA-QC/Datasets/R204brafv600e/Masks"
    OUTPUT_PATH="/home/pouya/Develop/UBC/QA-QC/Datasets/R204brafv600e/Results"

    # processor = SlideProcessor(args.slides_path, args.masks_path, args.output_path, args.down_sample_rate)
    processor = SlideProcessor(SLIDE_PATH, MASKS_PATH, OUTPUT_PATH, 5)
    processor.process_slides()

if __name__ == "__main__":
    main()