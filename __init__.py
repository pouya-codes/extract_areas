import glob, os
OPENSLIDE_PATH = r'D:/Develop/UBC/openslide/bin'
os.add_dll_directory(OPENSLIDE_PATH)
import openslide
import cv2
import json
from PIL import Image
from process_file import ImageProcessor
import numpy as np
from matplotlib.path import Path
from myparser import parse_args
from utils import process_annotation, process_mask
from generate_mask import MaskGenerator
from patch_extractor import PatchExtractor
import pyvips
import time

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

    def init_annotation_exporter(self, patch_size, positive_annotations_label):
        self.patch_exporter = PatchExtractor(patch_size, positive_annotations_label)

    def init_mask_generator(self, model_path):
        self.mask_generator = MaskGenerator(model_path)


    def init_image_processor(self, model_dir, tile_size = 256, overlay_down_sample_rate = 5, post_processing = True, gpu_ids=[]):
        self.image_processor = ImageProcessor(model_dir, tile_size, post_processing, gpu_ids)
        self.overlay_down_sample_rate = overlay_down_sample_rate

    def open_wsi_slide(self, slide_path):
        try:
            slide = pyvips.Image.new_from_file(slide_path)
            # slide = openslide.OpenSlide(slide_path)
            return slide
        except Exception as e:
            print(f"Cannot open {slide_path}\nError: {e}")
            return None

    def process_slides(self, save_regions = False):
        slides = [slide for ext in self.extensions for slide in glob.glob(os.path.join(self.slides_path, ext))]
        for slide_path in slides:
            print(f"Processing {slide_path}")
            slide = self.open_wsi_slide(slide_path)
            # slide = pyvips.Image.new_from_file(slide_path)
            file_name = os.path.basename(slide_path).split('.')[0]
            slide_dimensions = slide.width, slide.height
            regions = None
            # slide_dimensions = slide.dimensions

            if os.path.exists(os.path.join(self.output_path, file_name)):
                print(f"Skipping {file_name} as it already exists in the output path")
                continue

            os.makedirs(os.path.join(self.output_path, file_name), exist_ok=True)

            has_annotation, has_mask = False, False
            if self.masks_path:
                mask_path = glob.glob(os.path.join(self.masks_path, file_name + "*.png"))
                has_mask = len(mask_path) == 1
                mask_path = mask_path[0]

            if self.annotations_path:
                annotation_path = glob.glob(os.path.join(self.annotations_path, file_name + "*.txt"))
                has_annotation = len(annotation_path) == 1

            if not self.annotations_path and not has_annotation and not has_mask:
                if hasattr(self, 'mask_generator'):
                    print(f"Generating mask for {slide_path}")
                    mask, thumb = self.mask_generator.generate_mask(slide_path)
                    thumb.write_to_file(os.path.join(self.output_path, file_name, f"{file_name}_thumb.png"))
                    mask_path = os.path.join(self.output_path, file_name, f"{file_name}_mask.png")
                    if(mask is not None):
                        has_mask = True
                        cv2.imwrite(mask_path, mask)
                else:
                    print(f"No mask/annotation found for {slide_path}. Skipping...")
                    continue


            if hasattr(self, 'image_processor'):
                heat_map = np.zeros((slide_dimensions[1] // self.overlay_down_sample_rate, slide_dimensions[0] // self.overlay_down_sample_rate, 3), dtype=np.uint8)

            if has_annotation:
                annotation_path = annotation_path[0]
                regions = process_annotation(annotation_path)

            
            if has_mask:
                regions = process_mask(mask_path, slide_dimensions)
            
            if regions is None: # If no regions are found
                continue

            regions["Mask"] = regions.get("Mask", [])
            
            os.makedirs(os.path.join(self.output_path, file_name), exist_ok=True)

            for label, areas in regions.items():
                for area in areas:
                    x, y, width, height, *_ = area if len(area) == 5 else area + [None]
                    region = slide.crop(x, y, width, height)
                    region = Image.fromarray(region.numpy())
                    # region = slide.read_region((x, y), 0, (width, height))
                    if (save_regions):
                        os.makedirs(os.path.join(self.output_path, file_name, label), exist_ok=True)
                        img_path = os.path.join(self.output_path, file_name, label, f"{x}_{y}_{width}_{height}.png")
                        # region.write_to_file(img_path)
                        region.save(img_path)
                    # continue
                    if hasattr(self, 'image_processor'):

                        region = region.resize((width // self.slide_down_sample_rate, height // self.slide_down_sample_rate))
                        results, scoring = self.image_processor.test_img(region, eager_mode=True, color_dapi=True, color_marker=True)
                        overlay_image = results["SegRefined"]

                        if hasattr(self, 'patch_exporter'):
                            self.patch_exporter.export_patches(region, overlay_image, label , area, file_name)
                        
                        if (save_regions):
                            np_array = np.array(overlay_image)
                            background = region.convert("RGBA")
                            overlay = overlay_image.convert("RGBA")
                            new_img = Image.blend(background, overlay, 0.25)
                            img_path = os.path.join(self.output_path, file_name, label, f"{x}_{y}_{width}_{height}_overlaid.png")
                            new_img.save(img_path)
                            if scoring is not None:
                                json_path = os.path.join(self.output_path, file_name, label, f"{x}_{y}_{width}_{height}.json")
                                with open(json_path, 'w') as f:
                                    json.dump(scoring, f, indent=2)


                        x, y, width, height = ( x // self.overlay_down_sample_rate,
                                                y // self.overlay_down_sample_rate, 
                                                width // self.overlay_down_sample_rate, 
                                                height // self.overlay_down_sample_rate)
                        
                        width_overlay, height_overlay = overlay_image.size
                        overlay_to_slide_ratio = self.overlay_down_sample_rate // self.slide_down_sample_rate
                        np_array = np.array(overlay_image.resize((width_overlay //overlay_to_slide_ratio, height_overlay // overlay_to_slide_ratio)))
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
                os.makedirs(os.path.join(self.output_path, file_name), exist_ok=True)
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

    if args.mask_generator is not None:
        processor.init_mask_generator(args.model_path)

    if args.export_positive_annotations is not None:
        processor.init_annotation_exporter( 32, args.positive_annotations_label)

    processor.process_slides(save_regions=False)

if __name__ == "__main__":
    main()