import glob
import os
import gc

# OPENSLIDE_PATH = r'D:/Develop/UBC/openslide/bin'
# os.add_dll_directory(OPENSLIDE_PATH)
# import openslide
import cv2
import json
from PIL import Image
from src.process_file import ImageProcessor
import numpy as np
from matplotlib.path import Path
from myparser import parse_args
from src.utils import (
    process_annotation,
    process_mask,
    process_qupath_dearray,
    create_mask_from_annotation,
    create_tissue_mask,
)
from src.generate_mask import MaskGenerator
from src.patch_extractor import PatchExtractor
from src.patch_classifier import PatchClassifier
from src.read_metadata import ReadMetadataReader
import pyvips
import time
from tqdm import tqdm
from COI import COIs


class SlideProcessor:
    def __init__(
        self,
        slides_path,
        output_path,
        masks_path="",
        annotations_path="",
        qupath_dearray_paths="",
        slide_down_sample_rate=5,
        overlay_down_sample_rate=4,
    ):
        self.slides_path = slides_path
        self.output_path = output_path
        self.masks_path = masks_path
        self.annotations_path = annotations_path
        self.qupath_dearray_paths = qupath_dearray_paths
        self.slide_down_sample_rate = slide_down_sample_rate
        self.overlay_down_sample_rate = overlay_down_sample_rate
        self.extensions = ["*.svs", "*.tiff", "*.tif", "*.ndpi"]
        self.gradcam = False

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def init_annotation_exporter(
        self, patch_size, positive_annotations_label, output_path
    ):
        self.patch_exporter = PatchExtractor(
            patch_size, output_path, positive_annotations_label, window_size=32
        )

    def init_mask_generator(self, model_path, model_config):
        self.mask_generator = MaskGenerator(model_path, model_config)

    def init_image_processor(
        self, model_dir, tile_size=256, post_processing=True, gpu_ids=[]
    ):
        self.image_processor = ImageProcessor(
            model_dir, tile_size, post_processing, gpu_ids
        )

    def init_patch_classifier(self, model_path, patch_size=64):
        # self.patch_classifier = CellClassifier(model_path, generate_gradcam = True if staining == 'membrane' else False) # generate gradcam if membrane staining
        self.gradcam = False
        self.patch_classifier = PatchClassifier(
            model_path, patch_size=patch_size, generate_gradcam=self.gradcam
        )

    def open_wsi_slide(self, slide_path):
        try:
            slide = pyvips.Image.new_from_file(slide_path)
            return slide
        except Exception as e:
            print(f"Cannot open {slide_path}\nError: {e}")
            return None

    def init_metadata(self, metadata_path, metadata_sheet, dearry_map_file):
        self.metadata_reader = ReadMetadataReader(
            metadata_path, metadata_sheet, dearry_map_file
        )

    def process_slides(self, staining, save_regions=False, save_json=True):
        slides = [
            slide
            for ext in self.extensions
            for slide in glob.glob(os.path.join(self.slides_path, ext))
        ]
        for slide_path in slides:
            print(f"Processing {slide_path}")
            slide = self.open_wsi_slide(slide_path)
            # slide = pyvips.Image.new_from_file(slide_path)
            file_name = os.path.basename(slide_path).split(".")[0]
            slide_dimensions = slide.width, slide.height
            regions = None
            # slide_dimensions = slide.dimensions

            # if os.path.exists(os.path.join(self.output_path, file_name)):
            #     print(f"Skipping {file_name} as it already exists in the output path")
            #     continue

            os.makedirs(os.path.join(self.output_path, file_name), exist_ok=True)

            has_annotation, has_mask, has_dearray = False, False, False
            if self.masks_path:
                if os.path.exists(os.path.join(self.masks_path, file_name + ".png")):
                    mask_path = os.path.join(self.masks_path, file_name + ".png")
                    has_mask = True
                else:
                    mask_path = glob.glob(
                        os.path.join(self.masks_path, file_name + "*.png")
                    )
                    has_mask = len(mask_path) == 1
                    mask_path = mask_path[0]

            if self.annotations_path:
                annotation_path = glob.glob(
                    os.path.join(self.annotations_path, f"*{file_name}.txt")
                )
                has_annotation = len(annotation_path) == 1

            if self.qupath_dearray_paths:
                dearray_path = glob.glob(
                    os.path.join(self.qupath_dearray_paths, file_name + "*.txt")
                )
                has_dearray = len(dearray_path) == 1

            if (
                not has_annotation
                and not has_annotation
                and not has_mask
                and not has_dearray
            ):
                if hasattr(self, "mask_generator"):
                    mask_path = os.path.join(
                        self.output_path, file_name, f"{file_name}_mask.png"
                    )
                    has_mask = os.path.exists(mask_path)
                    if not has_mask:
                        mask, thumb = self.mask_generator.generate_mask(slide_path)
                        thumb.write_to_file(
                            os.path.join(
                                self.output_path, file_name, f"{file_name}_thumb.png"
                            )
                        )

                        if mask is not None:
                            print(f"{mask_path}")
                            has_mask = True
                            cv2.imwrite(mask_path, mask)
                            exit(0)
                else:
                    print(f"No mask/annotation found for {slide_path}. Skipping...")
                    continue

            if hasattr(self, "image_processor") or hasattr(self, "patch_classifier"):
                heat_map = np.zeros(
                    (
                        slide_dimensions[1] // self.overlay_down_sample_rate,
                        slide_dimensions[0] // self.overlay_down_sample_rate,
                        3,
                    ),
                    dtype=np.uint8,
                )

            if hasattr(self, "patch_classifier"):
                if self.gradcam:
                    gradcam_overlay = np.zeros(
                        (
                            slide_dimensions[1] // self.overlay_down_sample_rate,
                            slide_dimensions[0] // self.overlay_down_sample_rate,
                            3,
                        ),
                        dtype=np.uint8,
                    )

            if has_annotation:
                annotation_path = annotation_path[0]
                regions = process_annotation(annotation_path)

            if has_mask:
                regions = process_mask(mask_path, slide_dimensions)

            if has_dearray:
                dearray_path = dearray_path[0]
                regions = process_qupath_dearray(
                    dearray_path,
                    slide,
                    1500,
                    (
                        self.metadata_reader.get_dearray_mapping()
                        if hasattr(self, "metadata_reader")
                        else None
                    ),
                )

            if regions is None:  # If no regions are found
                continue

            # regions["Mask"] = regions.get("Mask", [])
            qupath_exists = any(key.startswith("QuPath") for key in regions.keys())

            if qupath_exists and hasattr(self, "metadata_reader"):
                if self.metadata_reader.get_number_of_cores() != len(regions.items()):
                    print(
                        f"Warning: Number of cores in metadata {self.metadata_reader.get_number_of_cores()} does not match number of regions {len(regions.items())}"
                    )
                    # continue
                if not self.metadata_reader.check_slide_exists(file_name):
                    print(f"Warning: Slide {file_name} not found in metadata")
                    # continue
            if self.qupath_dearray_paths and not qupath_exists:
                continue

            os.makedirs(os.path.join(self.output_path, file_name), exist_ok=True)

            for label, areas in (
                regions.items() if not qupath_exists else tqdm(regions.items())
            ):
                # core_label = file_name.split('_')[0] + "_" + label.split(',')[1]
                # if core_label not in COIs:
                #     continue
                # if label == "Other" or label == "Stroma":
                # continue
                for area in areas if qupath_exists else tqdm(areas):
                    x, y, width, height, *area_path = (
                        area if len(area) == 5 else area + [None]
                    )
                    # print(area_path[0])
                    # exit()
                    if area_path:
                        tissue_mask = create_mask_from_annotation(
                            x, y, width, height, area_path, self.slide_down_sample_rate
                        )
                        # a.save("test.png")
                        # exit()
                    else:
                        tissue_mask = None
                    # print(f"Processing {label} area {x}, {y}, {width}, {height}")
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if x + width > slide_dimensions[0]:
                        width = slide_dimensions[0] - x
                    if y + height > slide_dimensions[1]:
                        height = slide_dimensions[1] - y

                    region = slide.crop(x, y, width, height)
                    region = Image.fromarray(region.numpy())

                    if save_regions or save_json:
                        os.makedirs(
                            os.path.join(self.output_path, file_name, label),
                            exist_ok=True,
                        )
                    if save_regions:
                        os.makedirs(
                            os.path.join(self.output_path, file_name, label),
                            exist_ok=True,
                        )
                        img_path = os.path.join(
                            self.output_path,
                            file_name,
                            label,
                            f"{x}_{y}_{width}_{height}.png",
                        )
                        region_resized = region.resize(
                            (
                                int(region.width // self.overlay_down_sample_rate),
                                int(region.height // self.overlay_down_sample_rate),
                            )
                        )
                        region_resized.save(img_path)
                    # continue
                    if hasattr(self, "patch_exporter"):
                        self.patch_exporter.export_patches(
                            region, None, label, area, file_name
                        )

                    if hasattr(self, "image_processor") or hasattr(
                        self, "patch_classifier"
                    ):
                        region = region.resize(
                            (
                                width // self.slide_down_sample_rate,
                                height // self.slide_down_sample_rate,
                            )
                        )
                        # if area_path:
                        #     tissue_mask = create_tissue_mask(
                        #         x, y, width, height, area_path, self.slide_down_sample_rate
                        #     )
                        # else:
                        #     tissue_mask = None
                        if hasattr(self, "patch_classifier") and staining in [
                            "nuclear",
                            "cytoplasm",
                        ]:
                            gradcam_img, patch_classifier_mask, scoring = (
                                self.patch_classifier.process_image_with_sliding_window_batch(
                                    region, area
                                )
                            )
                            if save_regions:
                                patch_classifier_mask.save(
                                    os.path.join(
                                        self.output_path,
                                        file_name,
                                        label,
                                        f"{x}_{y}_{width}_{height}_classifier.png",
                                    )
                                )
                                # exit()
                                # gradcam.save(os.path.join(self.output_path, file_name, label, f"{x}_{y}_{width}_{height}_gradcam.png"))

                        if staining in ["nuclear", "cytoplasm"]:
                            results, scoring = self.image_processor.test_img(
                                region,
                                eager_mode=True,
                                color_dapi=True,
                                color_marker=True,
                                patch_classifier_mask=(
                                    patch_classifier_mask
                                    if (hasattr(self, "patch_classifier"))
                                    else None
                                ),
                                tissue_mask=tissue_mask,
                            )
                            overlay_image = results["SegRefined"]

                            if hasattr(self, "patch_exporter"):
                                cells_coords = scoring["cell_coords"]
                                self.patch_exporter.export_patches(
                                    region, cells_coords, label, area, file_name
                                )
                            else:
                                del scoring["cell_coords"]

                        elif staining == "membrane":
                            if hasattr(self, "patch_classifier"):
                                overlay_image, _, scoring = (
                                    self.patch_classifier.process_image_with_sliding_window_batch(
                                        region, area
                                    )
                                )
                            else:
                                print("Cell classifier not found. Skipping...")
                                continue

                        if scoring is not None:
                            overlay = np.array(overlay_image)
                            cv2.putText(
                                overlay,
                                f"Pos: {scoring['num_pos']}",
                                (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                overlay,
                                f"Neg: {scoring['num_neg']}",
                                (50, 300),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            if (
                                "QuPath" in label
                                and hasattr(self, "metadata_reader")
                                and self.metadata_reader.get_number_of_cores()
                                == len(regions.items())
                            ):
                                core_number = label.split(",")[1]
                                metadata = self.metadata_reader.get_metadata(
                                    file_name, core_number
                                )
                                if metadata is not None:
                                    cv2.putText(
                                        overlay,
                                        f"Label: {metadata}",
                                        (50, 400),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        4,
                                        (255, 255, 255),
                                        2,
                                        cv2.LINE_AA,
                                    )
                            overlay_image = Image.fromarray(overlay)

                        if save_regions:
                            np_array = np.array(overlay_image)
                            background = region.convert("RGBA")
                            overlay = overlay_image.convert("RGBA")
                            result_img = Image.blend(background, overlay, 0.25)
                            img_path = os.path.join(
                                self.output_path,
                                file_name,
                                label,
                                f"{x}_{y}_{width}_{height}_overlaid.png",
                            )
                            result_img_resized = result_img.resize(
                                (
                                    int(
                                        result_img.width
                                        // self.overlay_down_sample_rate
                                    ),
                                    int(
                                        result_img.height
                                        // self.overlay_down_sample_rate
                                    ),
                                )
                            )
                            result_img_resized.save(img_path)
                        if save_json:
                            if scoring is not None:
                                if "cell_coords" in scoring:
                                    del scoring["cell_coords"]
                                json_path = os.path.join(
                                    self.output_path,
                                    file_name,
                                    label,
                                    f"{x}_{y}_{width}_{height}.json",
                                )
                                with open(json_path, "w") as f:
                                    json.dump(scoring, f, indent=2)

                        x, y, width, height = (
                            x // self.overlay_down_sample_rate,
                            y // self.overlay_down_sample_rate,
                            width // self.overlay_down_sample_rate,
                            height // self.overlay_down_sample_rate,
                        )

                        width_overlay, height_overlay = overlay_image.size
                        overlay_to_slide_ratio = (
                            self.overlay_down_sample_rate // self.slide_down_sample_rate
                        )

                        np_array = np.array(
                            overlay_image.resize(
                                (
                                    width_overlay // overlay_to_slide_ratio,
                                    height_overlay // overlay_to_slide_ratio,
                                )
                            )
                        )
                        region_y, region_x = np_array.shape[0], np_array.shape[1]

                        region_height, region_width = y + region_y, x + region_x
                        offset_height, offset_width = 0, 0
                        if region_height > heat_map.shape[0]:
                            print(
                                f"Warning cropping region due to height {region_height} > {heat_map.shape[0]}"
                            )
                            offset_height = region_height - heat_map.shape[0]
                        if region_width > heat_map.shape[1]:
                            print(
                                f"Warning cropping region due to width {region_width} > {heat_map.shape[1]}"
                            )
                            offset_width = region_width - heat_map.shape[1]
                        # Overwrite only where heat_map has 0 values
                        heat_map_mask = (
                            heat_map[y:region_height, x:region_width, :] == 0
                        )
                        heat_map[y:region_height, x:region_width, :][heat_map_mask] = (
                            np_array[
                                : region_y - offset_height, : region_x - offset_width, :
                            ][heat_map_mask]
                        )
                        # heat_map[y:region_height, x:region_width, :] = np_array[
                        #     : region_y - offset_height, : region_x - offset_width, :
                        # ]
                        if self.gradcam and staining in ["nuclear", "cytoplasm"]:
                            np_array = np.array(
                                gradcam_img.resize(
                                    (
                                        width_overlay // overlay_to_slide_ratio,
                                        height_overlay // overlay_to_slide_ratio,
                                    )
                                )
                            )
                            gradcam_overlay[y:region_height, x:region_width, :] = (
                                np_array[
                                    : region_y - offset_height,
                                    : region_x - offset_width,
                                    :,
                                ]
                            )
                gc.collect()

            if hasattr(self, "image_processor"):
                if hasattr(
                    self, "metadata_reader"
                ) and self.metadata_reader.get_number_of_cores() != len(
                    regions.items()
                ):
                    text = self.metadata_reader.get_metadata_string(file_name)
                    for idx, line in enumerate(text.split("\n")):
                        cv2.putText(
                            heat_map,
                            line,
                            (40, 20 + (idx * 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                final_heat_map = Image.fromarray(heat_map)
                os.makedirs(os.path.join(self.output_path, file_name), exist_ok=True)
                final_heat_map.save(
                    os.path.join(self.output_path, file_name, f"{file_name}.png")
                )

                if self.gradcam:
                    Image.fromarray(gradcam_overlay).save(
                        os.path.join(
                            self.output_path, file_name, f"{file_name}_gradcam.png"
                        )
                    )


def main():
    args = parse_args()
    processor = SlideProcessor(
        args.slides_path,
        args.output_path,
        args.masks_path,
        args.annotations_path,
        args.qupath_dearray_paths,
        args.slide_down_sample_rate,
        args.overlay_down_sample_rate,
    )
    if args.deepliif:
        processor.init_image_processor(
            args.model_dir, args.tile_size, args.post_processing
        )

    if args.mask_generator:
        processor.init_mask_generator(args.model_path, args.model_config)

    if args.export_positive_annotations:
        processor.init_annotation_exporter(48, args.annotation_labels, args.output_path)

    if args.patch_classifier:
        processor.init_patch_classifier(
            args.patch_classifier_model, patch_size=args.patch_size
        )

    if args.metadata:
        processor.init_metadata(
            args.metadata_path, args.metadata_sheet, args.dearry_map_file
        )

    processor.process_slides(args.staining, save_regions=False, save_json=False)


if __name__ == "__main__":
    main()
