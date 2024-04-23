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

class SlideProcessor:
    def __init__(self, slides_path, masks_path, output_path, down_sample_rate):
        self.slides_path = slides_path
        self.masks_path = masks_path
        self.output_path = output_path
        self.down_sample_rate = down_sample_rate
        self.extensions = ['*.svs', '*.tiff', "*.tif", '*.ndpi']

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
            mask_path = glob.glob(os.path.join(self.masks_path, os.path.basename(slide_path).split('.')[0] + "*.png"))
            if len(mask_path) != 1:
                print(f"No/Multiple masks found for {slide_path}")
                continue
            mask_path = mask_path[0]
            slide_dimensions =  slide.dimensions
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            ratio_width = round(slide_dimensions[0] / mask.shape[1])
            ratio_height = round(slide_dimensions[1] / mask.shape[0])
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for idx, contour in enumerate(contours):
                x, y, width, height = map(int, cv2.boundingRect(contour))
                # Apply the ratios to the coordinates and dimensions
                x, y, width, height = x * ratio_width, y * ratio_height, width * ratio_width, height * ratio_height
                region = slide.read_region((x, y), 0, (width, height)).resize((width // self.down_sample_rate, height // self.down_sample_rate))
                region.save(os.path.join(self.output_path, f"{os.path.basename(slide_path).split('.')[0]}_{idx+1}_region.png"))


def main():
    args = parse_args()
    processor = SlideProcessor(args.slides_path, args.masks_path, args.output_path, args.down_sample_rate)
    processor.process_slides()

if __name__ == "__main__":
    main()