import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract areas from WSI slides based on the provided HistoQC masks")
    parser.add_argument("--slides_path", type=str, help="Path to the WSI slides location", required=True)
    parser.add_argument("--masks_path", type=str, help="Path to the HistoQC masks location", required=False)
    parser.add_argument("--annotations_path", type=str, help="Path to the slides' annotation.", required=False)
    parser.add_argument("--qupath_dearray_paths", type=str, help="Path to the text files contain qupath dearray info", required=False)
    parser.add_argument("--output_path", type=str, help="Path to the output location", required=True)
    parser.add_argument("--slide_down_sample_rate", type=int, help="the rate of down sampling the extracted regions", default=5)
    parser.add_argument("--overlay_down_sample_rate", type=int, help="the rate of down sampling to generated overlay", default=5)
    parser.add_argument("--save_images", action='store_true', help="Save the individual area images for the processed tissues")

    # Add deepliif arguments
    parser.add_argument("--deepliif", action='store_true', help="Run deepliif")
    parser.add_argument("--model_dir", type=str, help="Path to the unserilized model directory")
    parser.add_argument("--tile_size", type=int, help="Size of the tiles to be processed", default=256)
    parser.add_argument("--post_processing", type=bool, help="Run postprocessing alogrithem on the results", default=True)

    # Add mask generation arguments
    parser.add_argument("--mask_generator", action='store_true', help="Run mask generator")
    parser.add_argument("--model_path", type=str, help="Path to the segment anything model checkpoint")

    # Export areas of positive annotations
    parser.add_argument("--export_positive_annotations", action='store_true', help="Export areas of positive annotations")
    parser.add_argument("--annotation_labels", type=str, nargs='+', help="The lable of the annotations to be exported", default=None)

    parser.add_argument("--patch_classifier", action='store_true', help="Apply cell classifier on the extracted cells")
    parser.add_argument("--patch_classifier_model", type=str, help="Path to the cell classifier model")

    # Add required staining argument
    parser.add_argument("--staining", type=str, choices=['nuclear', 'membrane', 'cytoplasm'], help="Type of staining", required=False, default='nuclear')

    # Add metadata argument"
    parser.add_argument("--metadata", action='store_true', help="Providing metadata", required=False)
    parser.add_argument("--metadata_path", type=str, help="Path to the excel metadata file", required=False, default=None)
    parser.add_argument("--metadata_sheet", type=str, help="Name of the sheet to read from the metadata file", required=False, default=None)
    parser.add_argument("--dearry_map_file", type=str, help="The path to csv file that contains dearry maping", required=False, default=None)

    return parser.parse_args()

