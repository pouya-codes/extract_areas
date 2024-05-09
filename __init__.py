import glob, os
OPENSLIDE_PATH = r'C:/Users/pouya/Develop/QA-QC/openslide/bin'
os.add_dll_directory(OPENSLIDE_PATH)
import openslide
import cv2
import re
from PIL import Image
from process_file import ImageProcessor
import numpy as np
from matplotlib.path import Path
from myparser import parse_args
from utils import process_annotation, process_mask

class SlideProcessor:
    def __init__(self, slides_path, output_path, masks_path = "", annotations_path = "", slide_down_sample_rate = 5):
        self.slides_path = slides_path
        self.output_path = output_path
        self.masks_path = masks_path
        self.annotations_path = annotations_path
        self.slide_down_sample_rate = slide_down_sample_rate
        self.extensions = ['*.svs', '*.tiff', "*.tif", '*.ndpi']

        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def init_image_processor(self, model_dir, tile_size = 512, overlay_down_sample_rate = 5, post_processing = True, gpu_ids=[]):
        self.image_processor = ImageProcessor(model_dir, tile_size, post_processing, gpu_ids)
        self.overlay_down_sample_rate = overlay_down_sample_rate

    def open_wsi_slide(self, slide_path):
        try:
            slide = openslide.OpenSlide(slide_path)
            return slide
        except Exception as e:
            print(f"Cannot open {slide_path}\nError: {e}")
            return None

    def process_slides(self, save_regions = False):
        slides = [slide for ext in self.extensions for slide in glob.glob(os.path.join(self.slides_path, ext))]
        for slide_path in slides:
            slide = self.open_wsi_slide(slide_path)
            file_name = os.path.basename(slide_path).split('.')[0]
            slide_dimensions = slide.dimensions

            has_annotation, has_mask = False, False
            if self.masks_path:
                mask_path = glob.glob(os.path.join(self.masks_path, file_name + "*.png"))
                has_mask = len(mask_path) == 1
            if self.annotations_path:
                annotation_path = glob.glob(os.path.join(self.annotations_path, file_name + "*.txt"))
                has_annotation = len(annotation_path) == 1

            if not has_annotation and not has_mask:
                print(f"No mask/annotation found for {slide_path}")
                continue


            if hasattr(self, 'image_processor'):
                heat_map = np.zeros((slide_dimensions[1] // self.overlay_down_sample_rate, slide_dimensions[0] // self.overlay_down_sample_rate, 3), dtype=np.uint8)

            if has_annotation:
                annotation_path = annotation_path[0]
                regions = process_annotation(annotation_path)

            
            if has_mask:
                mask_path = mask_path[0]
                regions = process_mask(mask_path, slide_dimensions)

            regions["Mask"] = regions.get("Mask", [])
            for label, areas in regions.items():
                for area in areas:
                    x, y, width, height = area
                    region = slide.read_region((x, y), 0, (width, height))
                    if (save_regions):
                        os.makedirs(os.path.join(self.output_path, file_name, label), exist_ok=True)
                        img_path = os.path.join(self.output_path, file_name, label, f"{x}_{y}_{width}_{height}.png")
                        region.save(img_path)
                    
                    if hasattr(self, 'image_processor'):

                        region = region.resize((width // self.slide_down_sample_rate, height // self.slide_down_sample_rate))
                        results = self.image_processor.test_img(region)
                        overlay_channel = results["SegRefined"]
                        np_array = np.array(overlay_channel)
                        
                        if (True):
                            background = region.convert("RGBA")
                            overlay = overlay_channel.convert("RGBA")
                            new_img = Image.blend(background, overlay, 0.25)
                            img_path = os.path.join(self.output_path, file_name, label, f"{x}_{y}_{width}_{height}_overlaid.png")
                            new_img.save(img_path)


                        x, y, width, height = ( x // self.overlay_down_sample_rate,
                                                y // self.overlay_down_sample_rate, 
                                                width // self.overlay_down_sample_rate, 
                                                height // self.overlay_down_sample_rate)

                        region_y, region_x = np_array.shape[0], np_array.shape[1]

                        region_height, region_width = y + region_y, x + region_x
                        offset_height, offset_width = 0, 0
                        if (region_height > heat_map.shape[0]):
                            print(f"Warning cropping region due to height {region_height} > {heat_map.shape[0]}")
                            offset_height = region_height - heat_map.shape[0]
                        if (region_width > heat_map.shape[1]):
                            print(f"Warning cropping region due to width {region_width} > {heat_map.shape[1]}")
                            offset_width = region_width - heat_map.shape[1]
                            
                        heat_map[y:region_height, x:region_width, :] = np_array[:region_y-offset_height, :region_x-offset_width, :]


            if hasattr(self, 'image_processor'):
                final_heat_map = Image.fromarray(heat_map)
                final_heat_map.save(os.path.join(self.output_path, file_name, f"{file_name}.png"))


def main():
    args = parse_args()
    processor = SlideProcessor(args.slides_path, 
                               args.output_path,
                               args.masks_path, 
                               args.annotations_path,
                               args.slide_down_sample_rate)
    if args.deepliif is not None:
        processor.init_image_processor(args.model_dir, args.tile_size, args.overlay_down_sample_rate, args.post_processing)
    processor.process_slides(save_regions=True)

if __name__ == "__main__":
    main()